# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from functools import partial

import numpy as np

from skbio.diversity._util import (
    _validate_counts_matrix,
    vectorize_counts_and_tree,
)
from skbio.diversity._phylogenetic import _tip_distances
from skbio.util._decorator import params_aliased
from skbio.tree._utils import _validate_taxa_and_tree


# The default value indicating whether normalization should be applied
# for weighted UniFrac. This is used in two locations, so set in a single
# variable to avoid the code base becoming out of sync in the event of a
# change in this default value.
_normalize_weighted_unifrac_by_default = False


@params_aliased([("taxa", "otu_ids", "0.6.0", True)])
def unweighted_unifrac(u_counts, v_counts, taxa, tree, validate=True):
    """Compute unweighted UniFrac.

    Parameters
    ----------
    u_counts, v_counts: list, np.array
        Vectors of counts/abundances of taxa for two samples. Must be equal
        length.
    taxa : list, np.array
        Vector of taxon IDs corresponding to tip names in ``tree``. Must be the
        same length as ``u_counts`` and ``v_counts``. Required.
    tree : skbio.TreeNode
        Tree relating taxa. The set of tip names in the tree can be a superset
        of ``taxa``, but not a subset. Required.
    validate: bool, optional
        If ``False``, validation of the input won't be performed. This step can
        be slow, so if validation is run elsewhere it can be disabled here.
        However, invalid input data can lead to invalid results or error
        messages that are hard to interpret, so this step should not be
        bypassed if you're not certain that your input data are valid. See
        :mod:`skbio.diversity` for the description of what validation entails
        so you can determine if you can safely disable validation.

    Returns
    -------
    float
        The unweighted UniFrac distance between the two samples.

    Raises
    ------
    ValueError, MissingNodeError, DuplicateNodeError
        If validation fails. Exact error will depend on what was invalid.

    See Also
    --------
    weighted_unifrac
    skbio.diversity
    skbio.diversity.beta_diversity

    Notes
    -----
    Unweighted UniFrac was originally described in [1]_. A discussion of
    unweighted (qualitative) versus weighted (quantitative) diversity metrics
    is presented in [2]_. Deeper mathematical discussions of this metric is
    presented in [3]_.

    If computing unweighted UniFrac for multiple pairs of samples, using
    ``skbio.diversity.beta_diversity`` will be much faster than calling this
    function individually on each sample.

    This implementation differs from that in PyCogent (and therefore QIIME
    versions less than 2.0.0) by imposing a few additional restrictions on the
    inputs. First, the input tree must be rooted. In PyCogent, if an unrooted
    tree was provided that had a single trifurcating node (a newick convention
    for unrooted trees) that node was considered the root of the tree. Next,
    all taxa must be tips in the tree. PyCogent would silently ignore taxa that
    were not present the tree. To reproduce UniFrac results from PyCogent with
    scikit-bio, ensure that your PyCogent UniFrac calculations are performed on
    a rooted tree and that all taxa are present in the tree.

    This implementation of unweighted UniFrac is the array-based implementation
    described in [4]_.

    If using large number of samples or a large tree, we advise using the
    optimized UniFrac library [5]_.

    References
    ----------
    .. [1] Lozupone, C. & Knight, R. UniFrac: a new phylogenetic method for
       comparing microbial communities. Appl. Environ. Microbiol. 71, 8228-8235
       (2005).

    .. [2] Lozupone, C. A., Hamady, M., Kelley, S. T. & Knight, R. Quantitative
       and qualitative beta diversity measures lead to different insights into
       factors that structure microbial communities. Appl. Environ. Microbiol.
       73, 1576-1585 (2007).

    .. [3] Lozupone, C., Lladser, M. E., Knights, D., Stombaugh, J. & Knight,
       R. UniFrac: an effective distance metric for microbial community
       comparison. ISME J. 5, 169-172 (2011).

    .. [4] Hamady M, Lozupone C, Knight R. Fast UniFrac: facilitating high-
       throughput phylogenetic analyses of microbial communities including
       analysis of pyrosequencing and PhyloChip data.  ISME J. 4(1):17-27
       (2010).

    .. [5] https://github.com/biocore/unifrac

    Examples
    --------
    Assume we have the following abundance data for two samples, ``u`` and
    ``v``, represented as a pair of counts vectors. These counts represent the
    number of times specific Operational Taxonomic Units, or taxa, were
    observed in each of the samples.

    >>> u_counts = [1, 0, 0, 4, 1, 2, 3, 0]
    >>> v_counts = [0, 1, 1, 6, 0, 1, 0, 0]

    Because UniFrac is a phylogenetic diversity metric, we need to know which
    taxon each count corresponds to, which we'll provide as ``taxa``.

    >>> taxa = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8']

    We also need a phylogenetic tree that relates the taxa to one another.

    >>> from io import StringIO
    >>> from skbio import TreeNode
    >>> tree = TreeNode.read(StringIO(
    ...                      '(((((U1:0.5,U2:0.5):0.5,U3:1.0):1.0):0.0,'
    ...                      '(U4:0.75,(U5:0.5,((U6:0.33,U7:0.62):0.5'
    ...                      ',U8:0.5):0.5):0.5):1.25):0.0)root;'))

    We can then compute the unweighted UniFrac distance between the samples.

    >>> from skbio.diversity.beta import unweighted_unifrac
    >>> uu = unweighted_unifrac(u_counts, v_counts, taxa, tree)
    >>> print(round(uu, 2))
    0.37

    """
    u_node_counts, v_node_counts, _, _, tree_index = _setup_pairwise_unifrac(
        u_counts, v_counts, taxa, tree, validate, normalized=False, unweighted=True
    )
    return _unweighted_unifrac(u_node_counts, v_node_counts, tree_index["length"])


@params_aliased([("taxa", "otu_ids", "0.6.0", True)])
def weighted_unifrac(
    u_counts,
    v_counts,
    taxa,
    tree,
    normalized=_normalize_weighted_unifrac_by_default,
    validate=True,
):
    """Compute weighted UniFrac with or without branch length normalization.

    Parameters
    ----------
    u_counts, v_counts: list, np.array
        Vectors of counts/abundances of taxa for two samples. Must be equal
        length.
    taxa : list, np.array
        Vector of taxon IDs corresponding to tip names in ``tree``. Must be the
        same length as ``u_counts`` and ``v_counts``. Required.
    tree : skbio.TreeNode
        Tree relating taxa. The set of tip names in the tree can be a superset
        of ``taxa``, but not a subset. Required.
    normalized: boolean, optional
        If ``True``, apply branch length normalization, which is described in
        [1]_. Resulting distances will then be in the range ``[0, 1]``.
    validate: bool, optional
        If ``False``, validation of the input won't be performed. This step can
        be slow, so if validation is run elsewhere it can be disabled here.
        However, invalid input data can lead to invalid results or error
        messages that are hard to interpret, so this step should not be
        bypassed if you're not certain that your input data are valid. See
        :mod:`skbio.diversity` for the description of what validation entails
        so you can determine if you can safely disable validation.

    Returns
    -------
    float
        The weighted UniFrac distance between the two samples.

    Raises
    ------
    ValueError, MissingNodeError, DuplicateNodeError
        If validation fails. Exact error will depend on what was invalid.

    See Also
    --------
    unweighted_unifrac
    skbio.diversity
    skbio.diversity.beta_diversity

    Notes
    -----
    Weighted UniFrac was originally described in [1]_, which includes a
    discussion of unweighted (qualitative) versus weighted (quantitiative)
    diversity metrics. Deeper mathemtical discussions of this metric is
    presented in [2]_.

    If computing weighted UniFrac for multiple pairs of samples, using
    ``skbio.diversity.beta_diversity`` will be much faster than calling this
    function individually on each sample.

    This implementation differs from that in PyCogent (and therefore QIIME
    versions less than 2.0.0) by imposing a few additional restrictions on the
    inputs. First, the input tree must be rooted. In PyCogent, if an unrooted
    tree was provided that had a single trifurcating node (a newick convention
    for unrooted trees) that node was considered the root of the tree. Next,
    all taxa must be tips in the tree. PyCogent would silently ignore taxa that
    were not present the tree. To reproduce UniFrac results from PyCogent with
    scikit-bio, ensure that your PyCogent UniFrac calculations are performed on
    a rooted tree and that all taxa are present in the tree.

    This implementation of weighted UniFrac is the array-based implementation
    described in [3]_.

    If using large number of samples or a large tree, we advise using the
    optimized UniFrac library [4]_.

    References
    ----------
    .. [1] Lozupone, C. A., Hamady, M., Kelley, S. T. & Knight, R. Quantitative
       and qualitative beta diversity measures lead to different insights into
       factors that structure microbial communities. Appl. Environ. Microbiol.
       73, 1576-1585 (2007).

    .. [2] Lozupone, C., Lladser, M. E., Knights, D., Stombaugh, J. & Knight,
       R. UniFrac: an effective distance metric for microbial community
       comparison. ISME J. 5, 169-172 (2011).

    .. [3] Hamady M, Lozupone C, Knight R. Fast UniFrac: facilitating high-
       throughput phylogenetic analyses of microbial communities including
       analysis of pyrosequencing and PhyloChip data.  ISME J. 4(1):17-27
       (2010).

    .. [4] https://github.com/biocore/unifrac

    Examples
    --------
    Assume we have the following abundance data for two samples, ``u`` and
    ``v``, represented as a pair of counts vectors. These counts represent the
    number of times specific taxa were observed in each of the samples.

    >>> u_counts = [1, 0, 0, 4, 1, 2, 3, 0]
    >>> v_counts = [0, 1, 1, 6, 0, 1, 0, 0]

    Because UniFrac is a phylogenetic diversity metric, we need to know which
    taxon each count corresponds to, which we'll provide as ``taxa``.

    >>> taxa = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8']

    We also need a phylogenetic tree that relates the taxa to one another.

    >>> from io import StringIO
    >>> from skbio import TreeNode
    >>> tree = TreeNode.read(StringIO(
    ...                      '(((((U1:0.5,U2:0.5):0.5,U3:1.0):1.0):0.0,'
    ...                      '(U4:0.75,(U5:0.5,((U6:0.33,U7:0.62):0.5'
    ...                      ',U8:0.5):0.5):0.5):1.25):0.0)root;'))

    Compute the weighted UniFrac distance between the samples.

    >>> from skbio.diversity.beta import weighted_unifrac
    >>> wu = weighted_unifrac(u_counts, v_counts, taxa, tree)
    >>> print(round(wu, 2))
    1.54

    Compute the weighted UniFrac distance between the samples including
    branch length normalization so the value falls in the range ``[0.0, 1.0]``.

    >>> wu = weighted_unifrac(u_counts, v_counts, taxa, tree, normalized=True)
    >>> print(round(wu, 2))
    0.33

    """
    (
        u_node_counts,
        v_node_counts,
        u_total_count,
        v_total_count,
        tree_index,
    ) = _setup_pairwise_unifrac(
        u_counts,
        v_counts,
        taxa,
        tree,
        validate,
        normalized=normalized,
        unweighted=False,
    )
    branch_lengths = tree_index["length"]

    if normalized:
        tip_indices = _get_tip_indices(tree_index)
        node_to_root_distances = _tip_distances(branch_lengths, tree, tip_indices)
        return _weighted_unifrac_normalized(
            u_node_counts,
            v_node_counts,
            u_total_count,
            v_total_count,
            branch_lengths,
            node_to_root_distances,
        )
    else:
        return _weighted_unifrac(
            u_node_counts, v_node_counts, u_total_count, v_total_count, branch_lengths
        )[0]


def _setup_pairwise_unifrac(
    u_counts, v_counts, taxa, tree, validate, normalized, unweighted
):
    if validate:
        counts = _validate_counts_matrix([u_counts, v_counts], cast_int=False)
        if counts.shape[1] != len(taxa):
            raise ValueError("`taxa` must be the same length as `counts`.")
        _validate_taxa_and_tree(taxa, tree, rooted=True, lengths=True)
    else:
        counts = np.vstack([u_counts, v_counts])
    counts_by_node, tree_index, branch_lengths = vectorize_counts_and_tree(
        counts, taxa, tree
    )
    total_counts = counts.sum(axis=1)
    return (*counts_by_node, *total_counts, tree_index)


def _unweighted_unifrac(u_node_counts, v_node_counts, branch_lengths):
    """Calculate unweighted UniFrac distance between samples.

    Parameters
    ----------
    u_node_counts, v_node_counts : np.array
        Vectors indicating presence (value greater than zero) and absence
        (value equal to zero) of nodes in two samples, `u` and `v`. Order is
        assumed to be the same as in `branch_lengths`.
    branch_lengths : np.array
        Vector of branch lengths of all nodes (tips and internal nodes) in
        postorder representation of their tree.

    Returns
    -------
    float
        Unweighted UniFrac distance between samples.

    Notes
    -----
    The count vectors passed here correspond to all nodes in the tree, not
    just the tips.

    """
    unique_nodes = np.logical_xor(u_node_counts, v_node_counts)
    observed_nodes = np.logical_or(u_node_counts, v_node_counts)
    unique_branch_length = (branch_lengths * unique_nodes).sum()
    observed_branch_length = (branch_lengths * observed_nodes).sum()
    if observed_branch_length == 0.0:
        # handle special case to avoid division by zero
        return 0.0
    return unique_branch_length / observed_branch_length


def _weighted_unifrac(
    u_node_counts, v_node_counts, u_total_count, v_total_count, branch_lengths
):
    """Calculate weighted Unifrac distance between samples.

    Parameters
    ----------
    u_node_counts, v_node_counts : np.array
        Vectors indicating presence (value greater than zero) and absence
        (value equal to zero) of nodes in two samples, `u` and `v`. Order is
        assumed to be the same as in `branch_lengths`.
    u_total_count, v_total_count : int
        The sum of ``u_node_counts`` and ``v_node_counts`` vectors,
        respectively. This could be computed internally, but since this is a
        private method and the calling function has already generated these
        values, this saves an iteration over each of these vectors.
    branch_lengths : np.array
        Vector of branch lengths of all nodes (tips and internal nodes) in
        postorder representation of their tree.

    Returns
    -------
    float
        Weighted UniFrac distance between samples.
    np.array of float
        Proportional abundance of each node in tree in sample `u`
    np.array of float
        Proportional abundance of each node in tree in sample `v`

    """
    if u_total_count > 0:
        # convert to relative abundances if there are any counts
        u_node_proportions = u_node_counts / u_total_count
    else:
        # otherwise, we'll just do the computation with u_node_counts, which
        # is necessarily all zeros
        u_node_proportions = u_node_counts

    if v_total_count > 0:
        v_node_proportions = v_node_counts / v_total_count
    else:
        v_node_proportions = v_node_counts

    wu = (branch_lengths * np.absolute(u_node_proportions - v_node_proportions)).sum()
    return wu, u_node_proportions, v_node_proportions


def _weighted_unifrac_normalized(
    u_node_counts,
    v_node_counts,
    u_total_count,
    v_total_count,
    branch_lengths,
    node_to_root_distances,
):
    """Calculate weighted normalized UniFrac distance between samples.

    Parameters
    ----------
    u_node_counts, v_node_counts : np.array
         Vectors indicating presence (value greater than zero) and absence
         (value equal to zero) of nodes in two samples, `u` and `v`. Order is
         assumed to be the same as in `branch_lengths`.
    u_total_count, v_total_count : int
         The sum of ``u_node_counts`` and ``v_node_counts`` vectors,
         respectively. This could be computed internally, but since this is a
         private method and the calling function has already generated these
         values, this saves an iteration over each of these vectors.
    branch_lengths : np.array
        Vector of branch lengths of all nodes (tips and internal nodes) in
        postorder representation of their tree.
    node_to_root_distances : np.ndarray
        1D column vector of branch lengths in post order form. There should be
        positions in this vector for all nodes in the tree, but only tips
        should be non-zero.

    Returns
    -------
    float
        Normalized weighted UniFrac distance between samples.

    Notes
    -----
    The count vectors passed here correspond to all nodes in the tree, not
    just the tips.

    """
    if u_total_count == 0.0 and v_total_count == 0.0:
        # handle special case to avoid division by zero
        return 0.0
    u, u_node_proportions, v_node_proportions = _weighted_unifrac(
        u_node_counts, v_node_counts, u_total_count, v_total_count, branch_lengths
    )
    c = _weighted_unifrac_branch_correction(
        node_to_root_distances, u_node_proportions, v_node_proportions
    )

    return u / c


def _setup_multiple_unifrac(counts, taxa, tree, validate):
    if validate:
        _validate_taxa_and_tree(taxa, tree, rooted=True, lengths=True)

    counts_by_node, tree_index, branch_lengths = vectorize_counts_and_tree(
        counts, taxa, tree
    )

    return counts_by_node, tree_index, branch_lengths


def _setup_multiple_unweighted_unifrac(counts, taxa, tree, validate):
    r"""Create optimized pdist-compatible unweighted UniFrac function.

    Parameters
    ----------
    counts : 2D array_like of ints or floats
        Matrix containing count/abundance data where each row contains counts
        of observations in a given sample.
    taxa: list, np.array
        Vector of taxon IDs corresponding to tip names in ``tree``. Must be the
        same length as ``u_counts`` and ``v_counts``. These IDs do not need to
        be in tip order with respect to the tree.
    tree: skbio.TreeNode
        Tree relating taxa. The set of tip names in the tree can be a superset
        of ``taxa``, but not a subset.
    validate: bool, optional
        If ``False``, validation of the input won't be performed.

    Returns
    -------
    function
        Optimized pairwise unweighted UniFrac calculator that can be passed
        to ``scipy.spatial.distance.pdist``.
    2D np.array of ints, floats
        Counts of all nodes in ``tree``.

    """
    counts_by_node, _, branch_lengths = _setup_multiple_unifrac(
        counts, taxa, tree, validate
    )

    f = partial(_unweighted_unifrac, branch_lengths=branch_lengths)

    return f, counts_by_node


def _setup_multiple_weighted_unifrac(counts, taxa, tree, normalized, validate):
    r"""Create optimized pdist-compatible weighted UniFrac function.

    Parameters
    ----------
    counts : 2D array_like of ints or floats
        Matrix containing count/abundance data where each row contains counts
        of observations in a given sample.
    taxa : list, np.array
        Vector of taxon IDs corresponding to tip names in ``tree``. Must be the
        same length as ``u_counts`` and ``v_counts``. These IDs do not need to
        be in tip order with respect to the tree.
    tree : skbio.TreeNode
        Tree relating taxa. The set of tip names in the tree can be a superset
        of ``taxa``, but not a subset.
    normalized : bool
        If ``True``, output will be normalized.
    validate: bool, optional
        If ``False``, validation of the input won't be performed.

    Returns
    -------
    function
        Optimized pairwise unweighted UniFrac calculator that can be passed
        to ``scipy.spatial.distance.pdist``.
    2D np.array of ints, floats
        Counts of all nodes in ``tree``.

    """
    counts_by_node, tree_index, branch_lengths = _setup_multiple_unifrac(
        counts, taxa, tree, validate
    )
    tip_indices = _get_tip_indices(tree_index)

    if normalized:
        node_to_root_distances = _tip_distances(branch_lengths, tree, tip_indices)

        def f(u_node_counts, v_node_counts):
            u_total_count = np.take(u_node_counts, tip_indices).sum()
            v_total_count = np.take(v_node_counts, tip_indices).sum()
            u = _weighted_unifrac_normalized(
                u_node_counts,
                v_node_counts,
                u_total_count,
                v_total_count,
                branch_lengths,
                node_to_root_distances,
            )
            return u
    else:

        def f(u_node_counts, v_node_counts):
            u_total_count = np.take(u_node_counts, tip_indices).sum()
            v_total_count = np.take(v_node_counts, tip_indices).sum()
            u, _, _ = _weighted_unifrac(
                u_node_counts,
                v_node_counts,
                u_total_count,
                v_total_count,
                branch_lengths,
            )
            return u

    return f, counts_by_node


def _get_tip_indices(tree_index):
    tip_indices = np.array(
        [n.id for n in tree_index["id_index"].values() if n.is_tip()], dtype=np.intp
    )
    return tip_indices


def _weighted_unifrac_branch_correction(
    node_to_root_distances, u_node_proportions, v_node_proportions
):
    """Calculate weighted unifrac branch length correction.

    Parameters
    ----------
    node_to_root_distances : np.ndarray
        1D column vector of branch lengths in post order form. There should be
        positions in this vector for all nodes in the tree, but only tips
        should be non-zero.
    u_node_proportions, v_node_proportions : np.ndarray
        Proportional abundace of observations of all nodes in the tree in
        samples ``u`` and ``v``, respectively.
    u_total_count, v_total_count : float
        The sum of the observations in samples ``u`` and ``v``, respectively.

    Returns
    -------
    np.ndarray
        The corrected branch lengths

    """
    return (
        node_to_root_distances.ravel() * (u_node_proportions + v_node_proportions)
    ).sum()
