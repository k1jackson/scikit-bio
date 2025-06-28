from numbers import Integral
from warnings import warn

import numpy as np
import pandas as pd
from numpy import dot, hstack
from numpy.linalg import qr, svd
from scipy.linalg import eigh

from skbio.util.config._dispatcher import _ingest_array, _create_table, _create_table_1d
from ._principal_coordinate_analysis import _fsvd
from ._ordination_results import OrdinationResults
from ._utils import scale

def pca(
    X,
    method="eigh",
    number_of_dimensions=0,
    inplace=False,
    output_format=None
):
    r"""Perform Principal Component Analysis (PCA)

    Parameters
    ----------
    X : table_like
        :math:`n \times m, n \geq m` matrix of input data, where :math:`n` is the
        number of samples and :math:`m` is the number of features. See the `DataTable
        <https://scikit.bio/docs/dev/generated/skbio.util.config.html#the-datatable-type>`_
        type documentation for details.
    method : str, optional
        Matrix decomposition method to use. Default is "eigh" (eigendecomposition),
        which computes exact eigenvectors and eigenvalues for all dimensions. The
        alternate is "fsvd" (fast singular value decomposition), a heuristic that can
        compute only a given number of components.
    number_of_dimensions : int or float, optional
        Number of principle components. This number determines how many eigenvectors 
        and eigenvalues will be returned. If an integer is provided, the exact number 
        of components will be retained. If a float between 0 and 1, it represents the 
        fractional cumulative variance to be retained. Default is 0, which will return 
        all components.
    inplace : bool, optional
        If True, the input distance matrix will be centered in-place to reduce memory
        consumption, at the cost of losing the original distances. Default is False.
    output_format : optional
        Standard ``DataTable`` parameter. See the `DataTable <https://scikit.bio/
        docs/dev/generated/skbio.util.config.html#the-datatable-type>`_ type
        documentation for details.

    Returns
    -------
    OrdinationResults
        Object that stores the PCA results, including eigenvalues, the proportion
        explained by each of them, and transformed sample coordinates.

    Raises
    ------
    ValueError
        If there are less features than samples

    See Also
    --------
    OrdinationResults

    References
    ----------
    .. [1] Major portions of this code are repurpossed from ._principal_coordinate_analysis.py

    """
    data, sample_ids, feature_ids = _ingest_array(X)
    print(data.shape)
    n_samples, n_features = data.shape

    if n_samples < n_features:
      raise ValueError("Data cannot have fewer samples than features.")

    # Center data column-wise
    data = scale(data, with_std = False)

    if number_of_dimensions == 0:
        if method == "fsvd" and n_samples > 10:
            warn(
                "FSVD: since no value for number_of_dimensions is specified, "
                "PCA for all dimensions will be computed, which may result "
                "in long computation time if the number of samples is large.",
                RuntimeWarning,
            )
        number_of_dimensions = n_samples
    elif number_of_dimensions < 0:
        raise ValueError(
            "Invalid operation: cannot reduce distance matrix "
            "to negative dimensions using PCA. Did you intend "
            'to specify the default value "0", which sets '
            "the number_of_dimensions equal to the "
            "number of samples?"
        )
    elif not isinstance(number_of_dimensions, Integral) and number_of_dimensions > 1:
        raise ValueError(
            "Invalid operation: A floating-point number greater than 1 cannot be "
            "supplied as the number of dimensions."
        )

    cov = np.dot(data.T, data) / float(n_samples - 1)
    print(cov.shape)

    # Perform eigendecomposition
    if method == "eigh":
        eigvals, eigvecs = eigh(cov)        
        long_method_name = "Principal Component Analysis"
    elif method == "fsvd":
        number_dimensions = number_of_dimensions
        if 0 < number_of_dimensions < 1:
            warn(
                "FSVD: since value for number_of_dimensions is specified as float, "
                "PCA for all dimensions will be computed, which may result in long "
                "computation time if the number of samples is large. "
                "Consider specifying an integer value to optimize performance.",
                RuntimeWarning,
            )
            number_dimensions = n_samples
        eigenvals, eigvecs = _fsvd(cov, num_dimensions)
        long_method_name = long_method_name = "Approximate Principal Component Analysis using FSVD"
    else:
        raise ValueError(
            "PCA eigendecomposition method {} not supported.".format(method)
        )

    negative_close_to_zero = np.isclose(eigvals, 0)
    eigvals[negative_close_to_zero] = 0

    idxs_descending = eigvals.argsort()[::-1]
    eigvals = eigvals[idxs_descending]
    eigvecs = eigvecs[:, idxs_descending]
    
    num_positive = (eigvals >= 0).sum()
    eigvecs[:, num_positive:] = np.zeros(eigvecs[:, num_positive:].shape)
    eigvals[num_positive:] = np.zeros(eigvals[num_positive:].shape)

    if method == "fsvd":
        sum_eigenvalues = np.trace(data)
    else:
        sum_eigenvalues = np.sum(eigvals)

    proportion_explained = eigvals / sum_eigenvalues
    if 0 < number_of_dimensions < 1:
        cumulative_variance = np.cumsum(proportion_explained)
        num_dimensions = (
            np.searchsorted(cumulative_variance, number_of_dimensions, side="left") + 1
        )
        number_of_dimensions = num_dimensions

    eigvecs = eigvecs[:, :number_of_dimensions]
    eigvals = eigvals[:number_of_dimensions]
    proportion_explained = proportion_explained[:number_of_dimensions]
    print(eigvals.shape)
    print(eigvecs.shape)

    coordinates = eigvecs * np.sqrt(eigvals)

    axis_labels = ["PC%d" % i for i in range(1, number_of_dimensions + 1)]
    print(axis_labels.shape)
    return OrdinationResults(
        short_method_name = 'PCA',
        long_method_name = long_method_name,
        eigvals = _create_table_1d(eigvals, index = axis_labels, backend = output_format),
        samples=_create_table(
            coordinates,
            index=distance_matrix.ids,
            columns=axis_labels,
            backend=output_format,
        ),
        proportion_explained=_create_table_1d(
            proportion_explained, index=axis_labels, backend=output_format
        ),
    )