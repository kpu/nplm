#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/functional/hash.hpp>
#ifdef USE_CHRONO
#include <boost/chrono.hpp>
#endif

#include <Eigen/Dense>

#include "maybe_omp.h"

// Make matrices hashable

namespace Eigen {
    template <typename Derived>
    size_t hash_value(const DenseBase<Derived> &m)
    {
        size_t h=0;
	for (int i=0; i<m.rows(); i++)
	    for (int j=0; j<m.cols(); j++)
	        boost::hash_combine(h, m(i,j));
	return h;
    }
}

namespace nplm
{

void splitBySpace(const std::string &line, std::vector<std::string> &items);
void readWordsFile(std::ifstream &TRAININ, std::vector<std::string> &word_list);
void readWordsFile(const std::string &file, std::vector<std::string> &word_list);
void writeWordsFile(const std::vector<std::string> &words, std::ofstream &file);
void writeWordsFile(const std::vector<std::string> &words, const std::string &filename);
void readDataFile(const std::string &filename, int &ngram_size, std::vector<int> &data, int minibatch_size=0);
void readUnigramProbs(const std::string &unigram_probs_file, std::vector<double> &unigram_probs);
void readSentFile(const std::string &file, std::vector<std::vector<std::string> > &sentences);

// Functions that take non-const matrices as arguments
// are supposed to declare them const and then use this
// to cast away constness.
#define UNCONST(t,c,uc) Eigen::MatrixBase<t> &uc = const_cast<Eigen::MatrixBase<t>&>(c);

template <typename Derived>
void initMatrix(boost::random::mt19937 &engine,
		const Eigen::MatrixBase<Derived> &p_const,
		bool init_normal, double range)
{
    UNCONST(Derived, p_const, p);
    if (init_normal == 0)
     // initialize with uniform distribution in [-range, range]
    {
        boost::random::uniform_real_distribution<> unif_real(-range, range); 
        for (int i = 0; i < p.rows(); i++)
        {
            for (int j = 0; j< p.cols(); j++)
            {
                p(i,j) = unif_real(engine);    
            }
        }

    }
    else 
      // initialize with gaussian distribution with mean 0 and stdev range
    {
        boost::random::normal_distribution<double> unif_normal(0., range);
        for (int i = 0; i < p.rows(); i++)
        {
            for (int j = 0; j < p.cols(); j++)
            {
                p(i,j) = unif_normal(engine);    
            }
        }
    }
}

template <typename Derived>
void readMatrix(std::ifstream &TRAININ, Eigen::MatrixBase<Derived> &param_const)
{
    UNCONST(Derived, param_const, param);

    int i = 0;
    std::string line;
    std::vector<std::string> fields;
    
    while (std::getline(TRAININ, line) && line != "")
    {
        splitBySpace(line, fields);
	if (fields.size() != param.cols())
	{
	    std::ostringstream err;
	    err << "error: wrong number of columns (expected " << param.cols() << ", found " << fields.size() << ")";
	    throw std::runtime_error(err.str());
	}
	
	if (i >= param.rows())
	{
	    std::ostringstream err;
	    err << "error: wrong number of rows (expected " << param.rows() << ", found " << i << ")";
	    throw std::runtime_error(err.str());
	}
	
	for (int j=0; j<fields.size(); j++)
	{
	    param(i,j) = boost::lexical_cast<typename Derived::Scalar>(fields[j]);
	}
	i++;
    }
    
    if (i != param.rows())
    {
        std::ostringstream err;
	err << "error: wrong number of rows (expected " << param.rows() << ", found more)";
	throw std::runtime_error(err.str());
    }
}

template <typename Derived>
void readMatrix(const std::string &param_file, const Eigen::MatrixBase<Derived> &param_const)
{
    UNCONST(Derived, param_const, param);
    std::cerr << "Reading data from file: " << param_file << std::endl;
    
    std::ifstream TRAININ(param_file.c_str());
    if (!TRAININ)
    {
        std::cerr << "Error: can't read training data from file " << param_file << std::endl;
	exit(-1);
    }
    readMatrix(TRAININ, param);
    TRAININ.close();
}

template <typename Derived>
void writeMatrix(const Eigen::MatrixBase<Derived> &param, const std::string &filename)
{
    std::cerr << "Writing parameters to " << filename << std::endl;

    std::ofstream OUT;
    OUT.precision(16);
    OUT.open(filename.c_str());
    if (! OUT)
    {
      std::cerr << "Error: can't write to file " << filename<< std::endl;
      exit(-1);
    }
    writeMatrix(param, OUT);
    OUT.close();
}

template <typename Derived>
void writeMatrix(const Eigen::MatrixBase<Derived> &param, std::ofstream &OUT)
{
    for (int row = 0;row < param.rows();row++)
    {
        int col;
        for (col = 0;col < param.cols()-1;col++)
        {
            OUT<<param(row,col)<<"\t";
        }
        //dont want an extra tab at the end
        OUT<<param(row,col)<<std::endl;
    }
}

template <typename Derived>
double logsum(const Eigen::MatrixBase<Derived> &v)
{
    int mi; 
    double m = v.maxCoeff(&mi);
    double logz = 0.0;
    for (int i=0; i<v.rows(); i++)
        if (i != mi)
	    logz += std::exp(v(i) - m);
    logz = log1p(logz) + m;
    return logz;
}

double logadd(double x, double y);

#ifdef USE_CHRONO
class Timer 
{
    typedef boost::chrono::high_resolution_clock clock_type;
    typedef clock_type::time_point time_type;
    typedef clock_type::duration duration_type;
    std::vector<time_type> m_start;
    std::vector<duration_type> m_total;
public:
    Timer() { }
    Timer(int n) { resize(n); }
    void resize(int n) { m_start.resize(n); m_total.resize(n); }
    int size() const { return m_start.size(); }
    void start(int i);
    void stop(int i);
    void reset(int i);
    double get(int i) const;
};

extern Timer timer;
#define start_timer(x) timer.start(x)
#define stop_timer(x) timer.stop(x)
#else
#define start_timer(x) 0
#define stop_timer(x) 0
#endif

int setup_threads(int n_threads);

} // namespace nplm
