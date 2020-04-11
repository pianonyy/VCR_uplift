# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False



from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset

from libc.math cimport sqrt

import numpy as np

cimport numpy as np
np.import_array()

from ._utils cimport log
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    def __dealloc__(self):
        """Destructor."""

        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                   SIZE_t end) nogil:
        """Placeholder for a method which will initialize the criterion.

        Parameters
        ----------
        y: array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
        y_stride: SIZE_t
            y_stride is used to index the kth output value as follows:
            y[i, k] = y[i * y_stride + k]
        sample_weight: array-like, dtype=DOUBLE_t
            The weight of each sample
        weighted_n_samples: DOUBLE_t
            The total weight of the samples being considered
        samples: array-like, dtype=DOUBLE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start: SIZE_t
            The first sample to be used on this node
        end: SIZE_t
            The last sample used on this node

        """

        pass

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """

        pass

    cdef void reverse_reset(self) nogil:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    cdef void update(self, SIZE_t new_pos) nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos: SIZE_t
            New starting index position of the samples in the right child
        """

        pass

    cdef double node_impurity(self) nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of samples[start:end]. This is the
        primary function of the criterion class.
        """

        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of samples[start:pos] + the impurity
        of samples[pos:end].

        Parameters
        ----------
        impurity_left: double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right: double pointer
            The memory address where the impurity of the right child should be
            stored
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of samples[start:end] and save the value into dest.

        Parameters
        ----------
        dest: double pointer
            The memory address where the node value should be stored.
        """

        pass

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double impurity_improvement(self, double impurity) nogil:
        """Placeholder for improvement in impurity after a split.

        Placeholder for a method which computes the improvement
        in impurity when a split occurs. The weighted impurity improvement
        equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity: double
            The initial impurity of the node before the split

        Return
        ------
        double: improvement in impurity after the split occurs
        """

        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right /
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left /
                             self.weighted_n_node_samples * impurity_left)))


cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    cdef SIZE_t* n_classes
    cdef SIZE_t sum_stride
    cdef double p_value

    def __cinit__(self, SIZE_t n_outputs,
                  np.ndarray[SIZE_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs: SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes: numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """

        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Count labels for each output
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        self.n_classes = NULL

        safe_realloc(&self.n_classes, n_outputs)

        cdef SIZE_t k = 0
        cdef SIZE_t sum_stride = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > sum_stride:
                sum_stride = n_classes[k]

        self.sum_stride = sum_stride

        cdef SIZE_t n_elements = n_outputs * sum_stride
        self.sum_total = <double*> calloc(n_elements, sizeof(double))
        self.sum_left = <double*> calloc(n_elements, sizeof(double))
        self.sum_right = <double*> calloc(n_elements, sizeof(double))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""

        free(self.n_classes)

    def __reduce__(self):
        return (ClassificationCriterion,
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride,
                   DOUBLE_t* sample_weight, double weighted_n_samples,
                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
        children samples[start:start] and samples[start:end].

        Parameters
        ----------
        y: array-like, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency
        y_stride: SIZE_t
            The stride between elements in the buffer, important if there
            are multiple targets (multi-output)
        sample_weight: array-like, dtype=DTYPE_t
            The weight of each sample
        weighted_n_samples: SIZE_t
            The total weight of all samples
        samples: array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        start: SIZE_t
            The first sample to use in the mask
        end: SIZE_t
            The last sample to use in the mask
        """

        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        for k in range(self.n_outputs):
            memset(sum_total + offset, 0, n_classes[k] * sizeof(double))
            offset += self.sum_stride

        for p in range(start, end):
            i = samples[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0
            if sample_weight != NULL:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = <SIZE_t> y[i * y_stride + k]
                sum_total[k * self.sum_stride + c] += w

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""

        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_left, 0, n_classes[k] * sizeof(double))
            memcpy(sum_right, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride

    cdef void reverse_reset(self) nogil:
        """Reset the criterion at pos=end."""
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_right, 0, n_classes[k] * sizeof(double))
            memcpy(sum_left, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride

    cdef void update(self, SIZE_t new_pos) nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        Parameters
        ----------
        new_pos: SIZE_t
            The new ending position for which to move samples from the right
            child to the left child.
        """
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = (k * self.sum_stride +
                                   <SIZE_t> y[i * self.y_stride + k])
                    sum_left[label_index] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = (k * self.sum_stride +
                                   <SIZE_t> y[i * self.y_stride + k])
                    sum_left[label_index] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                sum_right[c] = sum_total[c] - sum_left[c]

            sum_right += self.sum_stride
            sum_left += self.sum_stride
            sum_total += self.sum_stride

        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest: double pointer
            The memory address which we will save the node value into.
        """

        cdef double* sum_total = self.sum_total
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memcpy(dest, sum_total, n_classes[k] * sizeof(double))
            dest += self.sum_stride
            sum_total += self.sum_stride

cdef class UpliftGini(ClassificationCriterion):


    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion."""

        cdef double* sum_total = self.sum_total
        cdef double impurity = 0.0
        cdef SIZE_t k

        cdef double n_c0
        cdef double n_c1
        cdef double n_t0
        cdef double n_t1

        cdef double n_t
        cdef double n_c

        cdef double p_t_1
        cdef double p_t_0
        cdef double p_c_1
        cdef double p_c_0

        for k in range(self.n_outputs):

            n_c0 = sum_total[0]
            n_c1 = sum_total[1]
            n_t0 = sum_total[2]
            n_t1 = sum_total[3]

            n_t = n_t0 + n_t1
            n_c = n_c0 + n_c1

            p_t_1 = (n_t1 + 0.5) / (n_t + 1)
            p_t_0 = 1 - p_t_1
            p_c_1 = (n_c1 + 0.5) / (n_c + 1)
            p_c_0 = 1 - p_c_1


            impurity += (p_t_1 - p_c_1)**2 + (p_t_0 - p_c_0)**2
            sum_total += self.sum_stride

        return impurity



    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double val_impurity_left = 0.0
        cdef double val_impurity_right = 0.0
        cdef SIZE_t k

        cdef double n_lc0
        cdef double n_lc1
        cdef double n_lt0
        cdef double n_lt1
        cdef double n_rc0
        cdef double n_rc1
        cdef double n_rt0
        cdef double n_rt1
        cdef double n_lt
        cdef double n_rt
        cdef double n_lc
        cdef double n_rc
        cdef double n_l
        cdef double n_r
        cdef double n

        cdef double p_lt_1
        cdef double p_lt_0
        cdef double p_lc_1
        cdef double p_lc_0
        cdef double p_rt_1
        cdef double p_rt_0
        cdef double p_rc_1
        cdef double p_rc_0
        cdef double p_l
        cdef double p_r

        for k in range(self.n_outputs):

            n_lc0 = self.sum_left[0]
            n_lc1 = self.sum_left[1]
            n_lt0 = self.sum_left[2]
            n_lt1 = self.sum_left[3]

            n_rc0 = self.sum_right[0]
            n_rc1 = self.sum_right[1]
            n_rt0 = self.sum_right[2]
            n_rt1 = self.sum_right[3]

            n_lt = n_lt1 + n_lt0
            n_rt = n_rt1 + n_rt0
            n_lc = n_lc1 + n_lc0
            n_rc = n_rc1 + n_rc0

            n_l = n_lt + n_lc
            n_r = n_rt + n_rc

            n = n_l + n_r

            p_lt_1 = (n_lt1 + 0.5) / (n_lt + 1)
            p_lt_0 = 1 - p_lt_1
            p_lc_1 = (n_lc1 + 0.5) / (n_lc + 1)
            p_lc_0 = 1 - p_lc_1
            p_rt_1 = (n_rt1 + 0.5) / (n_rt + 1)
            p_rt_0 = 1 - p_rt_1
            p_rc_1 = (n_rc1 + 0.5) / (n_rc + 1)
            p_rc_0 = 1 - p_rc_1

            p_l = (n_l + 0.5) / (n + 1)
            p_r = 1 - p_l


            val_impurity_left += p_l * ((p_lt_1 - p_lc_1)**2 + (p_lt_0 - p_lc_0)**2)
            val_impurity_right += p_r * ((p_rt_1 - p_rc_1)**2 + (p_rt_0 - p_rc_0)**2)

            sum_left += self.sum_stride
            sum_right += self.sum_stride


        impurity_left[0] = val_impurity_left
        impurity_right[0] = val_impurity_right

    cdef double proxy_impurity_improvement(self) nogil:


        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double impurity_improvement = 0.0
        cdef double impurity_left
        cdef double impurity_right
        cdef double count_k
        cdef SIZE_t k

        cdef double n_lc0
        cdef double n_lc1
        cdef double n_lt0
        cdef double n_lt1
        cdef double n_rc0
        cdef double n_rc1
        cdef double n_rt0
        cdef double n_rt1
        cdef double n_lt
        cdef double n_rt
        cdef double n_lc
        cdef double n_rc
        cdef double n_t
        cdef double n_c
        cdef double n

        cdef double KL_gain
        cdef double H
        cdef double KL
        cdef double H_t
        cdef double H_c
        cdef double I

        cdef double p_t_l
        cdef double p_t_r
        cdef double p_c_l
        cdef double p_c_r
        cdef double p_t
        cdef double p_c

        cdef double full_variance
        cdef double p_value

        cdef double react_rate_t
        cdef double react_rate_c
        cdef double variance_c
        cdef double variance_t

        for k in range(self.n_outputs):

            n_lc0 = self.sum_left[0]
            n_lc1 = self.sum_left[1]
            n_lt0 = self.sum_left[2]
            n_lt1 = self.sum_left[3]

            n_rc0 = self.sum_right[0]
            n_rc1 = self.sum_right[1]
            n_rt0 = self.sum_right[2]
            n_rt1 = self.sum_right[3]

            n_lt = n_lt1 + n_lt0
            n_rt = n_rt1 + n_rt0
            n_lc = n_lc1 + n_lc0
            n_rc = n_rc1 + n_rc0

            n_t = n_lt + n_rt
            n_c = n_lc + n_rc

            n = n_t + n_c

            p_t_l = (n_lt + 0.5) / (n_t + 1)
            p_t_r = 1 - p_t_l
            p_c_l = (n_lc + 0.5) / (n_c + 1)
            p_c_r = 1 - p_c_l

            p_t = (n_t + 0.5) / (n + 1)
            p_c = 1 - p_t

            #data manipulation stat tests

            #calculate for treatment group left node
            react_rate_t = (n_lt + 0.5) / (n_t + 0.1)
            variance_t = 0.0
            variance_t += (n_t * n_t * react_rate_t * (1 - react_rate_t)) / (n_lt * (n_t - n_lt) * (n_t - 1))
            #calculate for control group left node
            react_rate_c = (n_lc + 0.5) / (n_c + 0.1)
            variance_c = 0.0
            variance_c += (n_c * n_c * react_rate_c * (1 - react_rate_c)) / (n_lc * (n_c - n_lc) * (n_c - 1))
            #calculate full variance and p_value
            full_variance = variance_t + variance_c
            p_value = ((p_t_l - p_c_l) - (p_t_r - p_c_r)) / sqrt(full_variance)
            self.p_value = p_value


            # E_gain
            self.children_impurity(&impurity_left, &impurity_right)
            E_gain = impurity_left + impurity_right  - self.node_impurity()

            # J penalty
            Gini = p_t * p_c
            E = (p_t_l - p_c_l)**2 + (p_t_r - p_c_r)**2
            Gini_t = p_t * p_t_l * p_t_r
            Gini_c = p_c * p_c_l * p_c_r
            J = Gini * E + Gini_t + Gini_c + 0.5

            impurity_improvement += (E_gain / J)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        return impurity_improvement


cdef class UpliftRadcliffeSurryTSplit(ClassificationCriterion):
    """TODO
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion."""

        cdef double* sum_total = self.sum_total
        cdef double impurity = 0.0
        cdef SIZE_t k



        #имеем две группы: таргет с количеством n_t и контрол c количеством n_c
        #далее делим группы на подгруппы 0 и 1 с количеством (n_c0, n_t0) и (n_c1, n_t1) соответственно
        cdef double n_c0
        cdef double n_c1
        cdef double n_t0
        cdef double n_t1

        cdef double n_t
        cdef double n_c

        cdef double p_t_1
        cdef double p_t_0
        cdef double p_c_1
        cdef double p_c_0


        #имеем гипотезу p_t_0 - p_c_0 = p_t_1 - p_c_0
        for k in range(self.n_outputs):

            n_c0 = sum_total[0]
            n_c1 = sum_total[1]
            n_t0 = sum_total[2]
            n_t1 = sum_total[3]

            n_t = n_t0 + n_t1
            n_c = n_c0 + n_c1

            p_t_1 = (n_t1 + 0.5) / (n_t + 1)    ##зачем +0.5 и + 1 ???
            p_t_0 = 1 - p_t_1
            p_c_1 = (n_c1 + 0.5) / (n_c + 1)
            p_c_0 = 1 - p_c_1

            C_44 = (1 / n_t0) + (1 / n_t1) + (1/ n_c0) + (1 / n_c1)
            SSE = n_t0 * p_t_0 * (1 - p_t_0) + n_t1 * p_t_1 * (1 - p_t_1) + n_c0 * p_c_0 * (1 - p_c_0) + n_c1 * p_c_1 * (1 - p_c_1)

            impurity += ((n_t + n_c - 4)*( p_t_0 - p_c_0 - (p_t_1 - p_c_1))**2 ) / (C_44 * SSE)

            sum_total += self.sum_stride

        return impurity



    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double val_impurity_left = 0.0
        cdef double val_impurity_right = 0.0
        cdef SIZE_t k

        cdef double n_lc0
        cdef double n_lc1
        cdef double n_lt0
        cdef double n_lt1
        cdef double n_rc0
        cdef double n_rc1
        cdef double n_rt0
        cdef double n_rt1
        cdef double n_lt
        cdef double n_rt
        cdef double n_lc
        cdef double n_rc
        cdef double n_l
        cdef double n_r
        cdef double n

        cdef double p_lt_1
        cdef double p_lt_0
        cdef double p_lc_1
        cdef double p_lc_0
        cdef double p_rt_1
        cdef double p_rt_0
        cdef double p_rc_1
        cdef double p_rc_0
        cdef double p_l
        cdef double p_r

        for k in range(self.n_outputs):

            n_lc0 = self.sum_left[0]
            n_lc1 = self.sum_left[1]
            n_lt0 = self.sum_left[2]
            n_lt1 = self.sum_left[3]

            n_rc0 = self.sum_right[0]
            n_rc1 = self.sum_right[1]
            n_rt0 = self.sum_right[2]
            n_rt1 = self.sum_right[3]

            n_lt = n_lt1 + n_lt0
            n_rt = n_rt1 + n_rt0
            n_lc = n_lc1 + n_lc0
            n_rc = n_rc1 + n_rc0

            n_l = n_lt + n_lc
            n_r = n_rt + n_rc

            n = n_l + n_r

            p_lt_1 = (n_lt1 + 0.5) / (n_lt + 1)
            p_lt_0 = 1 - p_lt_1
            p_lc_1 = (n_lc1 + 0.5) / (n_lc + 1)
            p_lc_0 = 1 - p_lc_1
            p_rt_1 = (n_rt1 + 0.5) / (n_rt + 1)
            p_rt_0 = 1 - p_rt_1
            p_rc_1 = (n_rc1 + 0.5) / (n_rc + 1)
            p_rc_0 = 1 - p_rc_1

            p_l = (n_l + 0.5) / (n + 1)
            p_r = 1 - p_l

            val_impurity_left += p_l * ((p_lt_1 - p_lc_1)**2 + (p_lt_0 - p_lc_0)**2)
            val_impurity_right += p_r * ((p_rt_1 - p_rc_1)**2 + (p_rt_0 - p_rc_0)**2)

            sum_left += self.sum_stride
            sum_right += self.sum_stride


        impurity_left[0] = val_impurity_left
        impurity_right[0] = val_impurity_right

    cdef double proxy_impurity_improvement(self) nogil:
        """TODO
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double impurity_improvement = 0.0
        cdef double impurity_left
        cdef double impurity_right
        cdef double count_k
        cdef SIZE_t k

        cdef double n_lc0
        cdef double n_lc1
        cdef double n_lt0
        cdef double n_lt1
        cdef double n_rc0
        cdef double n_rc1
        cdef double n_rt0
        cdef double n_rt1
        cdef double n_lt
        cdef double n_rt
        cdef double n_lc
        cdef double n_rc
        cdef double n_t
        cdef double n_c
        cdef double n

        cdef double KL_gain
        cdef double H
        cdef double KL
        cdef double H_t
        cdef double H_c
        cdef double I

        cdef double p_t_l
        cdef double p_t_r
        cdef double p_c_l
        cdef double p_c_r
        cdef double p_t
        cdef double p_c

        for k in range(self.n_outputs):

            n_lc0 = self.sum_left[0]
            n_lc1 = self.sum_left[1]
            n_lt0 = self.sum_left[2]
            n_lt1 = self.sum_left[3]

            n_rc0 = self.sum_right[0]
            n_rc1 = self.sum_right[1]
            n_rt0 = self.sum_right[2]
            n_rt1 = self.sum_right[3]

            n_lt = n_lt1 + n_lt0
            n_rt = n_rt1 + n_rt0
            n_lc = n_lc1 + n_lc0
            n_rc = n_rc1 + n_rc0

            n_t = n_lt + n_rt
            n_c = n_lc + n_rc

            n = n_t + n_c

            p_t_l = (n_lt + 0.5) / (n_t + 1)
            p_t_r = 1 - p_t_l
            p_c_l = (n_lc + 0.5) / (n_c + 1)
            p_c_r = 1 - p_c_l

            p_t = (n_t + 0.5) / (n + 1)
            p_c = 1 - p_t


            # E_gain
            self.children_impurity(&impurity_left, &impurity_right)
            E_gain = impurity_left + impurity_right  - self.node_impurity()

            # J penalty
            Gini = p_t * p_c
            E = (p_t_l - p_c_l)**2 + (p_t_r - p_c_r)**2
            Gini_t = p_t * p_t_l * p_t_r
            Gini_c = p_c * p_c_l * p_c_r
            J = Gini * E + Gini_t + Gini_c + 0.5

            impurity_improvement += (E_gain / J)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        return impurity_improvement

cdef class UpliftEntropy(ClassificationCriterion):
    """TODO
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion."""

        cdef double* sum_total = self.sum_total
        cdef double impurity = 0.0
        cdef SIZE_t k

        cdef double n_c0
        cdef double n_c1
        cdef double n_t0
        cdef double n_t1

        cdef double n_t
        cdef double n_c

        cdef double p_t_1
        cdef double p_t_0
        cdef double p_c_1
        cdef double p_c_0

        for k in range(self.n_outputs):

            n_c0 = sum_total[0]
            n_c1 = sum_total[1]
            n_t0 = sum_total[2]
            n_t1 = sum_total[3]

            n_t = n_t0 + n_t1
            n_c = n_c0 + n_c1

            p_t_1 = (n_t1 + 0.5) / (n_t + 1)
            p_t_0 = 1 - p_t_1
            p_c_1 = (n_c1 + 0.5) / (n_c + 1)
            p_c_0 = 1 - p_c_1

            impurity += p_t_1 * log(p_t_1 / p_c_1) + p_t_0 * log(p_t_0 / p_c_0)
            sum_total += self.sum_stride

        return impurity





    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left: double pointer
            The memory address to save the impurity of the left node
        impurity_right: double pointer
            The memory address to save the impurity of the right node
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double val_impurity_left = 0.0
        cdef double val_impurity_right = 0.0
        cdef SIZE_t k

        cdef double n_lc0
        cdef double n_lc1
        cdef double n_lt0
        cdef double n_lt1
        cdef double n_rc0
        cdef double n_rc1
        cdef double n_rt0
        cdef double n_rt1
        cdef double n_lt
        cdef double n_rt
        cdef double n_lc
        cdef double n_rc
        cdef double n_l
        cdef double n_r
        cdef double n

        cdef double p_lt_1
        cdef double p_lt_0
        cdef double p_lc_1
        cdef double p_lc_0
        cdef double p_rt_1
        cdef double p_rt_0
        cdef double p_rc_1
        cdef double p_rc_0
        cdef double p_l
        cdef double p_r

        for k in range(self.n_outputs):

            n_lc0 = self.sum_left[0]
            n_lc1 = self.sum_left[1]
            n_lt0 = self.sum_left[2]
            n_lt1 = self.sum_left[3]

            n_rc0 = self.sum_right[0]
            n_rc1 = self.sum_right[1]
            n_rt0 = self.sum_right[2]
            n_rt1 = self.sum_right[3]

            n_lt = n_lt1 + n_lt0
            n_rt = n_rt1 + n_rt0
            n_lc = n_lc1 + n_lc0
            n_rc = n_rc1 + n_rc0

            n_l = n_lt + n_lc
            n_r = n_rt + n_rc

            n = n_l + n_r

            p_lt_1 = (n_lt1 + 0.5) / (n_lt + 1)
            p_lt_0 = 1 - p_lt_1
            p_lc_1 = (n_lc1 + 0.5) / (n_lc + 1)
            p_lc_0 = 1 - p_lc_1
            p_rt_1 = (n_rt1 + 0.5) / (n_rt + 1)
            p_rt_0 = 1 - p_rt_1
            p_rc_1 = (n_rc1 + 0.5) / (n_rc + 1)
            p_rc_0 = 1 - p_rc_1

            p_l = (n_l + 0.5) / (n + 1)
            p_r = 1 - p_l

            val_impurity_left += p_l * (p_lt_1 * log(p_lt_1 / p_lc_1) + p_lt_0 * log(p_lt_0 / p_lc_0))
            val_impurity_right += p_r * (p_rt_1 * log(p_rt_1 / p_rc_1) + p_rt_0 * log(p_rt_0 / p_rc_0))

            sum_left += self.sum_stride
            sum_right += self.sum_stride


        impurity_left[0] = val_impurity_left
        impurity_right[0] = val_impurity_right


    cdef double proxy_impurity_improvement(self) nogil:
        """TODO
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double impurity_improvement = 0.0
        cdef double impurity_left
        cdef double impurity_right
        cdef double count_k
        cdef SIZE_t k

        cdef double n_lc0
        cdef double n_lc1
        cdef double n_lt0
        cdef double n_lt1
        cdef double n_rc0
        cdef double n_rc1
        cdef double n_rt0
        cdef double n_rt1
        cdef double n_lt
        cdef double n_rt
        cdef double n_lc
        cdef double n_rc
        cdef double n_t
        cdef double n_c
        cdef double n

        cdef double KL_gain
        cdef double H
        cdef double KL
        cdef double H_t
        cdef double H_c
        cdef double I

        cdef double p_t_l
        cdef double p_t_r
        cdef double p_c_l
        cdef double p_c_r
        cdef double p_t
        cdef double p_c

        for k in range(self.n_outputs):

            n_lc0 = self.sum_left[0]
            n_lc1 = self.sum_left[1]
            n_lt0 = self.sum_left[2]
            n_lt1 = self.sum_left[3]

            n_rc0 = self.sum_right[0]
            n_rc1 = self.sum_right[1]
            n_rt0 = self.sum_right[2]
            n_rt1 = self.sum_right[3]

            n_lt = n_lt1 + n_lt0
            n_rt = n_rt1 + n_rt0
            n_lc = n_lc1 + n_lc0
            n_rc = n_rc1 + n_rc0

            n_t = n_lt + n_rt
            n_c = n_lc + n_rc

            n = n_t + n_c

            p_t_l = (n_lt + 0.5) / (n_t + 1)
            p_t_r = 1 - p_t_l
            p_c_l = (n_lc + 0.5) / (n_c + 1)
            p_c_r = 1 - p_c_l

            p_t = (n_t + 0.5) / (n + 1)
            p_c = 1 - p_t



            # KL_gain
            self.children_impurity(&impurity_left, &impurity_right)
            KL_gain = impurity_left + impurity_right  - self.node_impurity()

            # I penalty
            H = - p_t * log(p_t) - p_c * log(p_c)
            KL = p_t_l * log(p_t_l / p_c_l) + p_t_r * log(p_t_r / p_c_r)
            H_t = - p_t * (p_t_l * log(p_t_l) + p_t_r * log(p_t_r))
            H_c = - p_c * (p_c_l * log(p_c_l) + p_c_r * log(p_c_r))
            I = H * KL + H_t + H_c + 0.5

            impurity_improvement += (KL_gain / I)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        return impurity_improvement


cdef class Entropy(ClassificationCriterion):
    """Cross Entropy impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion."""

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double entropy = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)

            sum_total += self.sum_stride

        return entropy / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left: double pointer
            The memory address to save the impurity of the left node
        impurity_right: double pointer
            The memory address to save the impurity of the right node
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_left[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)

                count_k = sum_right[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = entropy_left / self.n_outputs
        impurity_right[0] = entropy_right / self.n_outputs


cdef class Gini(ClassificationCriterion):
    """Gini Index impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """



    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end] using the Gini criterion."""


        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double gini = 0.0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count = 0.0

            for c in range(n_classes[k]):
                count_k = sum_total[c]
                sq_count += count_k * count_k

            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)

            sum_total += self.sum_stride

        return gini / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left: DTYPE_t
            The memory address to save the impurity of the left node to
        impurity_right: DTYPE_t
            The memory address to save the impurity of the right node to
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double sq_count_left
        cdef double sq_count_right
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0

            for c in range(n_classes[k]):
                count_k = sum_left[c]
                sq_count_left += count_k * count_k

                count_k = sum_right[c]
                sq_count_right += count_k * count_k

            gini_left += 1.0 - sq_count_left / (self.weighted_n_left *
                                                self.weighted_n_left)

            gini_right += 1.0 - sq_count_right / (self.weighted_n_right *
                                                  self.weighted_n_right)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = gini_left / self.n_outputs
        impurity_right[0] = gini_right / self.n_outputs


cdef class RegressionCriterion(Criterion):
    """Abstract regression criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    cdef double sq_sum_total

    def __cinit__(self, SIZE_t n_outputs):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs: SIZE_t
            The number of targets to be predicted
        """

        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        # Allocate memory for the accumulators
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __reduce__(self):
        return (RegressionCriterion, (self.n_outputs,), self.__getstate__())

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                   SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        self.sq_sum_total = 0.0
        memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

    cdef void reverse_reset(self) nogil:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

    cdef void update(self, SIZE_t new_pos) nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef DOUBLE_t* y = self.y
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = y[i * self.y_stride + k]
                    sum_left[k] += w * y_ik

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = y[i * self.y_stride + k]
                    sum_left[k] -= w * y_ik

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""


        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = y[i * self.y_stride + k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


cdef class FriedmanMSE(MSE):
    """Mean squared error impurity criterion with improvement score by Friedman

    Uses the formula (35) in Friedmans original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0

        cdef SIZE_t k
        cdef double diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right)

        return diff * diff / (self.weighted_n_left * self.weighted_n_right)

    cdef double impurity_improvement(self, double impurity) nogil:
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0

        cdef SIZE_t k
        cdef double diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right) / self.n_outputs

        return (diff * diff / (self.weighted_n_left * self.weighted_n_right *
                               self.weighted_n_node_samples))
