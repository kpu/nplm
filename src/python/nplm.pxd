from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "neuralLM.h":
    cdef cppclass c_neuralLM "nplm::neuralLM":
        c_neuralLM()
        void set_normalization(bint)
        void set_map_digits(char)
        void set_log_base(double)
        void read(string filename) except +
        int get_order()
        int lookup_word(string)
        float lookup_ngram(vector[int])
        float lookup_ngram(int *, int)
        void set_cache(int)
        double cache_hit_rate()

cdef class NeuralLM:
    cdef c_neuralLM *thisptr
    cdef int c_lookup_word(self, char *s)
    cdef float c_lookup_ngram(self, int *words, int n)
    cdef readonly int order
    
