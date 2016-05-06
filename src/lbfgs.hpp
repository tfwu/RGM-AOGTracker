// This file is modified from FFLDv2 (the Fast Fourier Linear Detector version 2)
// Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>

#ifndef RGM_LBFGS_HPP_
#define RGM_LBFGS_HPP_

#include "util/UtilLog.hpp"

namespace RGM {

class LBFGS {
  public:
    /// Callback interface to provide objective function and gradient evaluations.
    class IFunction {
      public:
        /// Destructor.
        virtual ~IFunction();

        /// Returns the number of variables.
        virtual int dim() const = 0;

        /// Provides objective function and gradient evaluations.
        /// @param[in] x Current solution.
        /// @param[out] g The gradient vector which must be computed for the current solution.
        /// @returns The value of the objective function for the current solution.
        virtual double operator()(const double * x, double * g = 0) = 0; // const = 0;
    };

  public:
    /// Constructor.
    /// @param[in] function Callback function to provide objective function and gradient
    /// evaluations.
    /// @param[in] optTol Tolerance used to check for progress
    /// @param[in] suffDec
    /// @param[in] maxIterations Maximum number of iterations allowed.
    /// @param[in] correction
    /// @param[in] interp
    LBFGS(IFunction * function = 0, double optTol = 1e-6, double suffDec = 1e-4,
          int maxIterations = 1000, int correction = 100, int interp = 1);

    /// Starts the L-BFGS optimization process.
    /// @param[in,out] x Initial solution on entry. Receives the optimization result on exit.
    /// @returns The final value of the objective function.
    double operator()(double * x, const double * xLB, int method);

  private:
    /// Charles dubout's version
    double LBFGS1(double * argx, const double * xLB);

    /// based on minConf by Mark Schmidt  (http://www.di.ens.fr/~mschmidt/)
    /// which is also used in voc-releas5
    double LBFGS2(double * argx, const double * xLB);

  private:
    // Constructor parameters
    IFunction * function_;
    double optTol_;
    double suffDec_;
    int maxIterations_;
    int correction_;
    int interp_;

    int maxLineSearches_;
    int maxHistory_;

    DEFINE_RGM_LOGGER;
};

} // namespace RGM

#endif // RGM_LBFS_HPP_



