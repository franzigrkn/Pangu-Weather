import numpy as np


######### GraphCast cosine based area weighting #########

def deg2rad(deg: np.ndarray) -> np.ndarray:
    """Converts degrees to radians.

    Parameters
    ----------
    deg :
        Array of shape (N, ) containing the degrees

    Returns
    -------
    np.ndarray
        Array of shape (N, ) containing the radians
    """
    return deg * np.pi / 180

def get_lat_weights_cosine(lat: np.ndarray, unit="deg") -> np.ndarray:
    """Normalized area of the latitude-longitude grid cell"""
    if unit == "deg":
        lat = deg2rad(lat)
    area = np.abs(np.cos(lat))
    return area / np.mean(area)

######### WeatherBench sine based area weighting #########

def _assert_increasing(x: np.ndarray):
    if not (np.diff(x, axis=0) > 0).all():
        raise ValueError(f"array is not increasing: {x}")

def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
    pi_over_2 = np.pi / 2
    pi_over_2_array = np.full((1, x.shape[1]), pi_over_2, dtype=x.dtype)
    return np.concatenate([-pi_over_2_array, (x[:-1] + x[1:]) / 2, pi_over_2_array])

def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
    """Calculate the area overlap as a function of latitude."""
    bounds = _latitude_cell_bounds(points)
    _assert_increasing(bounds)
    upper = bounds[1:]
    lower = bounds[:-1]
    # normalized cell area: integral from lower to upper of cos(latitude)
    return np.sin(upper) - np.sin(lower)

def get_lat_weights_sine(lat: np.ndarray, unit="deg"):
    """Computes latitude/area weights from latitude coordinate of dataset."""
    weights = _cell_area_from_latitude(deg2rad(lat))
    weights /= np.mean(weights)

    return weights

class WeatherBenchLoss_plain:
    """
    Implementation of WeatherBench Loss function for pure evaluation when using only preds and gt.
    input dim: 4 x 721 x 1440
    target dim: 4 x 721 x 1440
    """
    def __init__(self):

        # latitude longitude grid
        input_res_lat = 721
        input_res_lon = 1440
        latitudes = np.linspace(-90, 90, num=input_res_lat)
        longitudes = np.linspace(-180, 180, num=input_res_lon + 1)[1:]
        lat_lon_grid = np.stack(np.meshgrid(latitudes, longitudes, indexing="ij"), axis=-1)

        # get area weighting
        self.area = get_lat_weights_sine(lat=lat_lon_grid[:, :, 0], unit='deg')
        #print(f"Shape of area weighting: {self.area.shape}")

    def compute_loss(self, input: np.ndarray, target: np.ndarray):
        """
        Computes the WeatherBench loss between input and target.

        Parameters
        ----------
        input : np.ndarray
            Predicted values with shape (variables, latitude, longitude).
        target : np.ndarray
            Ground truth values with shape (variables, latitude, longitude).

        Returns
        -------
        float
            The computed loss value.
        """
        #print(f"    Shape of input: {input.shape}")
        #print(f"    Shape of target: {target.shape}")
        # Loss calculation
        loss = (input - target) ** 2
        #print(f"    Shape of loss: {loss.shape}")
        # Scale by area
        #print(f"    Shape of area: {self.area.shape}")
        #loss = loss * self.area[np.newaxis, :, :]
        #print(f"    Shape of loss: {loss.shape}")
        # Mean + SQRT
        if len(loss.shape) == 3:
            #print(f"Shape of extended area: {self.area[np.newaxis, :, :].shape}")
            loss = loss * self.area[np.newaxis, :, :]
            reduce_dims = (1, 2)
        if len(loss.shape) == 4:
            #print(f"Shape of extended area: {self.area[np.newaxis, np.newaxis, :, :].shape}")
            loss = loss * self.area[np.newaxis, np.newaxis, :, :]
            reduce_dims = (2, 3)
        loss = np.sqrt(np.mean(loss, axis=reduce_dims))
        #loss = np.mean(loss, axis=reduce_dims)

        return loss