from typing import List, Optional, Tuple, Union, Any
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import torch
from torch import Tensor
from typing import Optional
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from .csr_operations import row_normalize

def sym_norm(edge_index: torch.Tensor,
             num_nodes: int,
             x: torch.Tensor, 
             coordinates: torch.Tensor,  # Node coordinates parameter
             m: int,                    # Azimuthal order parameter
             edge_weight: Optional[Union[Any,torch.Tensor]] = None,
             improved: Optional[bool] = False,
             dtype: Optional[Any] = None
             ) -> List:
    """
    Symmetric normalization with spatial coordinates
    
    Args:
        coordinates: Node coordinates tensor of shape [n, 2]
        edge_index: Edge index tensor of shape [2, E]
    """
    coor_arr = coordinates.cpu().numpy()
    edge_index_arr = edge_index.cpu().numpy()

    # Build CSR matrix with spatial distances
    start_points = edge_index_arr[0]
    end_points = edge_index_arr[1]
    num_nodes = np.max(edge_index_arr) + 1
    distances = np.linalg.norm(coor_arr[start_points] - coor_arr[end_points], axis=1)
    spatial_graph = csr_matrix((distances, (start_points, end_points)), shape=(num_nodes, num_nodes))

    # Row normalization
    graph_out = spatial_graph.copy()
    graph_out = row_normalize(graph_out, verbose=False)
    tensor_data_agf = torch.tensor(graph_out.data, dtype=torch.float32)

    if edge_weight is None:
        edge_weight = tensor_data_agf

    # Add self-loops if needed
    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    return edge_index, edge_weight

def azimuthal_gaussian_transform(
    x: Tensor,
    edge_index: Tensor,
    coor: Tensor,
    m: int,
    edge_weight: Optional[Tensor] = None,
    sigma: float = 1.0,
    rotation_factor: int = 2
) -> Tensor:
    """
    Azimuthal Gaussian Transform with complex weights and rotational phase differences
    
    Args:
        x: Node feature matrix [N, F]
        edge_index: Edge index [2, E]
        coor: Node coordinates [N, D] 
        m: Base harmonic order
        edge_weight: Predefined edge weights [E,]
        sigma: Gaussian kernel std
        rotation_factor: Rotation angle multiplier
    
    Returns:
        output: Direction-sensitive aggregated features [N, F]
    """
    row, col = edge_index
    num_nodes, feat_dim = x.shape

    # Compute Gaussian weights Γ_uv
    delta_coor = coor[col] - coor[row]  # Coordinate differences [E, D]
    distances = torch.norm(delta_coor, dim=1)  # [E,]
    
    if edge_weight is None:
        edge_weight = torch.exp(-distances**2 / (2 * sigma**2))  # [E,]

    # Compute complex phase
    phi = torch.atan2(delta_coor[:, 1], delta_coor[:, 0])  # Azimuthal angles [E,]
    phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)  # Convert to [0, 2π)
    
    base_phase = torch.exp(1j * m * phi)  # Base complex phase [E,]
    complex_weights = edge_weight * base_phase  # [E,]

    # Compute rotated phase
    rotated_phase = torch.exp(1j * m * rotation_factor * phi)  # [E,]
    rotated_weights = edge_weight * rotated_phase  # [E,]

    # Apply complex weights to neighbor features
    x_complex = x[col].to(torch.complex64)
    weighted_neigh = x_complex * complex_weights.view(-1, 1)  # [E, F]

    # Compute rotation weight difference
    rotated_weights_expanded = rotated_weights.view(-1, 1).expand(-1, feat_dim)
    diff_complex = weighted_neigh - rotated_weights_expanded  # [E, F]

    # Take modulus and aggregate
    diff_abs = torch.abs(diff_complex)  # [E, F]
    output = torch.zeros((num_nodes, feat_dim), dtype=x.dtype, device=x.device)
    output.index_add_(0, row, diff_abs)  # [N, F]

    return output

def gaussian_weight_1d(distance: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute 1D Gaussian weights"""
    sigma_tensor = torch.tensor(sigma, dtype=distance.dtype)
    two_pi_tensor = torch.tensor(2.0) * torch.pi
    return torch.exp(-0.5 * distance.pow(2) / sigma_tensor.pow(2)) / (sigma_tensor * torch.sqrt(two_pi_tensor))

class CombGaussian_F(MessagePassing):
    """
    Combined Gaussian Filter Convolution Layer
    
    Parameters:
        coordinates: Node coordinates tensor
        m: Azimuthal transform order
        K: Number of hops for propagation
        sigma: Gaussian standard deviation
    """
    def __init__(self, 
                 coordinates: torch.Tensor,
                 m: int,
                 K: int = 1, 
                 sigma: float = 1.0,
                 **kwargs):
        super(CombGaussian_F, self).__init__(aggr='add', **kwargs)
        self.coordinates = coordinates
        self.m = m
        self.K = K
        self.sigma = sigma

    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor, 
                coor: torch.Tensor,
                m: int,
                edge_weight: Optional[torch.Tensor] = None):
        xs = [x]
        # norm = None
        if m == 0:
            # Symmetric normalization mode
            edge_index, norm = sym_norm(edge_index, x.size(0), x, coor, m, edge_weight, dtype=x.dtype)
            for k in range(self.K):
                message = self.propagate(edge_index, x=xs[-1], norm=norm)
                xs.append(message)
        elif m > 0:
            # Azimuthal transform mode
            final_output = azimuthal_gaussian_transform(x, edge_index, coor, m)
            xs.append(final_output)
            
        return torch.cat(xs, dim=1) #,norm

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Message passing with Gaussian weights"""
        if self.m == 0:
            return x_j * gaussian_weight_1d(norm, self.sigma).view(-1, 1)
        elif self.m > 0:
            return x_j
    
    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

def theta_from_spatial_graph(locations: np.ndarray, spatial_graph: csr_matrix):
    """Compute azimuthal angles from spatial graph"""
    theta_data = np.zeros_like(spatial_graph.data, dtype=np.float32)

    for n in range(spatial_graph.indptr.shape[0] - 1):
        ptr_start, ptr_end = spatial_graph.indptr[n], spatial_graph.indptr[n + 1]
        nbr_indices = spatial_graph.indices[ptr_start:ptr_end]

        self_coord = locations[n, :]
        nbr_coord = locations[nbr_indices, :]
        relative_coord = nbr_coord - self_coord

        theta_data[ptr_start:ptr_end] = np.arctan2(relative_coord[:, 1], relative_coord[:, 0])

    theta_graph = spatial_graph.copy()
    theta_graph.data = theta_data

    return theta_graph

# (Optional)
def spatial_graph(edges: torch.Tensor):
    """Construct spatial graph from edge tensor"""
    start_points = edges[0].numpy()
    end_points = edges[1].numpy()
    num_nodes = max(np.max(start_points), np.max(end_points)) + 1
    data = np.ones(len(start_points))
    return csr_matrix((data, (start_points, end_points)), shape=(num_nodes, num_nodes))
