# distutils: language = c++

cdef class NeuralLM:
    def __cinit__(self, normalization=False, map_digits=None, cache_size=0):
        self.thisptr = new c_neuralLM()
        self.thisptr.set_normalization(normalization)
        self.thisptr.set_log_base(10.)
        if type(map_digits) is str and len(map_digits) == 1:
            self.thisptr.set_map_digits(map_digits)
        if cache_size:
            self.thisptr.set_cache(cache_size)

    def read(self, filename):
        self.thisptr.read(filename)
        self.order = self.thisptr.get_order()

    def get_order(self):
        return self.thisptr.get_order()

    def lookup_word(self, s):
        return self.thisptr.lookup_word(s)
    
    def lookup_ngram(self, words):
        if len(words) == 0:
            raise ValueError("ngram is empty")
        return self.thisptr.lookup_ngram(words)

    def cache_hit_rate(self):
        return self.thisptr.cache_hit_rate()

    # low-level interface that can be called by other Cython modules
    cdef int c_lookup_word(self, char *s):
        cdef string ss
        ss.assign(s)
        return self.thisptr.lookup_word(ss)

    cdef float c_lookup_ngram(self, int *words, int n):
        return self.thisptr.lookup_ngram(words, n)
