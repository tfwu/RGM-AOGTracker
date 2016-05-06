#include <Eigen/Core>

#include <algorithm>
#include <cassert>

#include "lbfgs.hpp"

namespace RGM {

LBFGS::IFunction::~IFunction() {
}


LBFGS::LBFGS(IFunction * function, double optTol, double suffDec,
             int maxIterations, int correction, int interp) : function_(function),
    optTol_(optTol), suffDec_(suffDec), maxIterations_(maxIterations),
    correction_(correction), interp_(interp), maxLineSearches_(40),
    maxHistory_(20) {

    RGM_CHECK_NOTNULL(function);
    RGM_CHECK_GT(function->dim(), 0);
    RGM_CHECK_GT(optTol, 0.0);
    RGM_CHECK_GT(suffDec, 0.0);
    RGM_CHECK_GT(maxIterations, 0);
    RGM_CHECK_GT(correction, 0);
    RGM_CHECK_GT(interp, 0);
}

double LBFGS::operator()(double * argx, const double * xLB, int method) {
    double f = 0;

    switch(method) {
        case 0:
            f = LBFGS1(argx, xLB);
            break;

        default:
            f = LBFGS2(argx, xLB);
            break;
    }

    return f;
}

double LBFGS::LBFGS1(double * argx, const double * xLB) {
    // Define the types ourselves to make sure that the matrices are col-major
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> VectorXd;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
    MatrixXd;

    RGM_CHECK_NOTNULL(function_);
    RGM_CHECK_NOTNULL(argx);

    int dim = function_->dim();

    // Convert the current solution to an Eigen::Map
    Eigen::Map<VectorXd> x(argx, dim);

    // project lower bound
    Eigen::Map<const VectorXd> lb(xLB, dim);

    x = x.cwiseMax(lb);

    // Initial value of the objective function and gradient
    VectorXd g(x.rows());
    double fx = (*function_)(x.data(), g.data());

    // Histories of the previous solution (required by L-BFGS)
    VectorXd px; // Previous solution x_{t-1}
    VectorXd pg; // Previous gradient g_{t-1}
    MatrixXd dxs(x.rows(),
                 maxHistory_); // History of the previous dx's = x_{t-1} - x_{t-2}, ...
    MatrixXd dgs(x.rows(),
                 maxHistory_); // History of the previous dg's = g_{t-1} - g_{t-2}, ...

    // Number of iterations remaining
    for(int i = 0, j = 0; j < maxIterations_; ++i, ++j) {
        // Relative tolerance
        const double relativeEpsilon = optTol_ * std::max(1.0, x.norm());

        // Check the norm of the gradient against convergence threshold
        if(g.norm() < relativeEpsilon) {
            return fx;
        }

        // Get a new descent direction using the L-BFGS algorithm
        VectorXd z = g;

        if(i && maxHistory_) {
            // Update the histories
            const int h = std::min(i, maxHistory_); // Current length of the histories
            const int end = (i - 1) % h;

            dxs.col(end) = x - px;
            dgs.col(end) = g - pg;

            // Initialize the variables
            VectorXd p(h);
            VectorXd a(h);

            for(int j = 0; j < h; ++j) {
                const int k = (end - j + h) % h;
                p(k) = 1.0 / dxs.col(k).dot(dgs.col(k));
                a(k) = p(k) * dxs.col(k).dot(z);
                z -= a(k) * dgs.col(k);
            }

            // Scaling of initial Hessian (identity matrix)
            z *= dxs.col(end).dot(dgs.col(end)) / dgs.col(end).dot(dgs.col(end));

            for(int j = 0; j < h; ++j) {
                const int k = (end + j + 1) % h;
                const double b = p(k) * dgs.col(k).dot(z);
                z += dxs.col(k) * (a(k) - b);
            }
        }

        // Save the previous state
        px = x;
        pg = g;

        // If z is not a valid descent direction (because of a bad Hessian estimation), restart the
        // optimization starting from the current solution
        double descent = -z.dot(g);

        if(descent > -0.0001 * relativeEpsilon) {
            z = g;
            i = 0;
            descent = -z.dot(g);
        }

        // Backtracking using Wolfe's first condition (Armijo condition)
        double step = i ? 1.0 : (1.0 / g.norm());
        bool down = false;
        int ls;

        for(ls = 0; ls < maxLineSearches_; ++ls) {
            // Tentative solution, gradient and loss
            const VectorXd nx = (x - step * z).cwiseMax(lb); //x - step * z
            VectorXd ng(x.rows());
            const double nfx = (*function_)(nx.data(), ng.data());

            if(nfx <= fx + 0.0001 * step * descent) {  // First Wolfe condition
                if((-z.dot(ng) >= 0.9 * descent) || down) {  // Second Wolfe condition
                    x = nx;
                    g = ng;
                    fx = nfx;
                    break;
                } else {
                    step *= 2.0;
                }
            } else {
                step *= 0.5;
                down = true;
            }
        }

        if(ls == maxLineSearches_) {
            if(i) {
                i = -1;
            } else {
                return fx;
            }
        }
    }

    return fx;
}

double LBFGS::LBFGS2(double * argx, const double * xLB) {
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrix;

    int dim = function_->dim();

    Eigen::Map<VectorXd> x(argx, dim);

    // project lower bounds
    Eigen::Map<const VectorXd> lb(xLB, dim);
    x = x.cwiseMax(lb);

    // compute gradient
    VectorXd grad(dim);
    double fx = (*function_)(x.data(), grad.data());

    // Compute Working Set
    VectorXd working(dim);
    working.fill(1.0F);

    for(int i = 0; i < dim; ++i) {
        if(x(i) < lb(i) + optTol_ * 2.0F && grad(i) >= 0) {
            working(i) = 0.0F;
        }
    }

    int numWorking = working.sum();

    VectorXd workingGrad(
        numWorking); // = (working.array()==0.0F).select(working, grad);
    for(int i = 0, j = 0; i < working.size(); ++i) {
        if(working(i) == 1.0F) {
            workingGrad(j++) = grad(i);
        }
    }

    if(numWorking == 0) {
        RGM_LOG(normal, "[LBFGS Optimization] All variables are at their bound and no further "
                "progress is possible at initial point");
        return fx;
    } else if(workingGrad.norm() <= optTol_) {
        RGM_LOG(normal, "[LBFGS Optimization] All working variables satisfy optimality condition "
                "at initial point");
        return fx;
    }

    int iter  = 0;
    int count = 0;
    VectorXd x_old, grad_old;
    Matrix  old_dirs(dim, correction_), old_stps(dim, correction_);
    double  Hdiag;

    int numCorrections = 0;
    while(iter < maxIterations_) {
        // Compute Step Direction
        VectorXd d = VectorXd::Zero(dim);

        if(count == 0) {
            VectorXd minus_grad = grad * -1;
            d = (working.array() != 0.0F).select(minus_grad, d);
            Hdiag = 1;
            numCorrections = 0;
        } else {
            // do lbfgs updating
            VectorXd y = grad - grad_old;
            VectorXd s = x - x_old;
            double ys  = (y.cwiseProduct(s)).sum();
            if(ys > 1e-10) {
                if(numCorrections < correction_) {
                    // full update
                    old_dirs.col(numCorrections) = s;
                    old_stps.col(numCorrections) = y;
                    numCorrections++;
                } else {
                    // limeted-memory update
                    Matrix tmpOld = old_dirs.block(0, 1, dim, correction_ - 1);
                    old_dirs.block(0, 0, dim, correction_ - 1).swap(tmpOld);
                    old_dirs.col(correction_ - 1) = s;

                    tmpOld = old_stps.block(0, 1, dim, correction_ - 1);
                    old_stps.block(0, 0, dim, correction_ - 1).swap(tmpOld);
                    old_stps.col(correction_ - 1) = y;
                }

                // Update scale of initial Hessian approximation
                Hdiag = ys / ((y.cwiseProduct(y)).sum());
            } else {
                RGM_LOG(normal, "[LBFGS Optimization] skipping lbfgs update");
            }

            VectorXd curvSat(numCorrections);
            curvSat.setZero();

            for(int j = 0; j < working.size(); ++j) {
                if(working(j) == 1.0F) {
                    curvSat += old_dirs.row(j).segment(0,
                                                       numCorrections).cwiseProduct(old_stps.row(j).segment(0, numCorrections));
                }
            }

            for(int j = 0; j < numCorrections; ++j) {
                curvSat(j) = (curvSat(j) > 1e-10) ? 1.0F : 0.0F;
            }

            int numStat = curvSat.sum();

            // BFGS Search Direction - get the (L-BFGS) approximate inverse Hessian multiplied by the gradient
            Matrix workingOldDirs(numWorking, numStat), workingOldStps(numWorking, numStat);

            for(int i = 0, i1 = 0; i < working.size(); ++i) {
                if(working(i) == 0.0F)
                    continue;
                for(int j = 0, j1 = 0; j < curvSat.size(); ++j) {
                    if(curvSat(j) == 0.0F)
                        continue;

                    workingOldDirs(i1, j1) = old_dirs(i, j);
                    workingOldStps(i1, j1) = old_stps(i, j);
                    j1++;
                }
                i1++;
            }


            VectorXd ro(numStat);
            ro.setZero();
            for(int i = 0; i < numStat; ++i) {
                ro(i) = 1.0F / (workingOldDirs.col(i).cwiseProduct(workingOldStps.col(
                                                                       i)).sum());
            }

            Matrix q = Matrix::Zero(numWorking, numStat + 1);
            Matrix r = Matrix::Zero(numWorking, numStat + 1);

            VectorXd al(numStat);
            al.setZero();

            VectorXd be(numStat);
            be.setZero();

            q.col(q.cols() - 1) = workingGrad * -1;

            for(int i = numStat - 1; i >= 0; --i) {
                al(i) = ro(i) * (workingOldDirs.col(i).cwiseProduct(q.col(i + 1)).sum());
                q.col(i) = q.col(i + 1) - al(i) * workingOldStps.col(i);
            }

            // Multiply by Initial Hessian
            r.col(0) = Hdiag * q.col(0);

            for(int i = 0; i < numStat; ++i) {
                be(i) = ro(i) * (workingOldStps.col(i).cwiseProduct(r.col(i)).sum());
                r.col(i + 1) = r.col(i) + workingOldDirs.col(i) * (al(i) - be(i));
            }

            for(int i = 0, j = 0; i < d.size(); ++i) {
                if(working(i) == 1.0F) {
                    d(i) = r(j++, r.cols() - 1);
                }
            }
        }

        grad_old = grad;
        x_old    = x;

        // Check that Progress can be made along the direction
        double fx_old = fx;
        double gtd = grad.cwiseProduct(d).sum();
        if(gtd > -optTol_) {
            RGM_LOG(normal, "[LBFGS Optimization] Directional derivative below optTol");
            break;
        }

        // Select Initial Guess to step length
        double t = 1.0F;
        if(count == 0) {
            t = std::min<double>(1.0F, 1.0F / workingGrad.cwiseAbs().sum());
        }

        // Evaluate the Objective and Projected Gradient at the Initial Step
        VectorXd x_tmp = x + t * d;
        VectorXd x_new = (x_tmp.array() < lb.array()).select(lb, x_tmp);


        VectorXd grad_new(dim);
        double fx_new = (*function_)(x_new.data(), grad_new.data());

        VectorXd grad_new_legal(dim);
        grad_new_legal.setZero();

        grad_new_legal = (grad_new.array() ==
                          std::numeric_limits<double>::quiet_NaN()).select(1, grad_new);
        grad_new_legal = (grad_new.array() ==
                          std::numeric_limits<double>::infinity()).select(1, grad_new);

        iter++;

        // Backtracking Line Search
        int lineSearchIters = 0;
        while(fx_new > fx + suffDec_ * grad.cwiseProduct(x_new - x).sum() ||
                (fx_new  == std::numeric_limits<double>::quiet_NaN() ||
                 fx_new  == std::numeric_limits<double>::infinity())) {
            double t_tmp = t;

            if(interp_ == 0 ||
                    (fx_new == std::numeric_limits<double>::quiet_NaN() ||
                     fx_new  == std::numeric_limits<double>::infinity()) ||
                    grad_new_legal.sum() > 0) {
                t *= 0.5F;
            } else {
                // [ 0  f      gtd;
                //   t  f_new  gtd_new ]
                double gtd_new = grad_new.cwiseProduct(d).sum();

                Matrix points(2, 3);
                points << 0, fx,     gtd,
                       t, fx_new, gtd_new;

                int minPos    = (t < 0) ? 1 : 0;
                int notMinPos = 1 - minPos;

                double d1 = points(minPos, 2) + points(notMinPos, 2)
                            - 3 * (points(minPos, 1) - points(notMinPos, 1)) / (points(minPos,
                                    0) - points(notMinPos, 0));

                double dtmp = d1 * d1 - points(minPos, 2) * points(notMinPos, 2);

                if(dtmp >= 0) {
                    double d2 = sqrt(dtmp);

                    double tt = points(notMinPos, 0) - (points(notMinPos, 0)
                                                        - points(minPos, 0)) * ((points(notMinPos, 2) + d2 - d1) / (points(notMinPos,
                                                                                2) - points(minPos, 2) + 2 * d2));

                    t = std::min<double>(std::max<double>(tt, points(minPos, 0)), points(notMinPos,
                                         0));
                } else {
                    t *= 0.5F;
                }
            }

            // Adjust if change is too small
            if(t < t_tmp * 1e-3) {
                t = t_tmp * 1e-3;
            } else if(t > t_tmp * 0.6F) {
                t = t_tmp * 0.6F;
            }

            // Check whether step has become too small
            if((t * d).cwiseAbs().sum() < optTol_) {
                RGM_LOG(warning, "[LBFGS Optimization] Line Search failed");
                t = 0;
                fx_new = fx;
                grad_new = grad;
                break;
            }

            // Evaluate New Point
            x_tmp = x + t * d;
            x_new = (x_tmp.array() < lb.array()).select(lb, x_tmp);

            fx_new = (*function_)(x_new.data(), grad_new.data());
            iter++;

            lineSearchIters++;
        } // while

        // Take step
        x = x_new;
        fx = fx_new;
        grad = grad_new;

        // Compute Working Set
        working.fill(1.0F);
        for(int i = 0; i < x.size(); ++i) {
            if(x(i) < lb(i) + optTol_ * 2 && grad(i) >= 0) {
                working(i) = 0.0F;
            }
        }

        numWorking = working.sum();
        workingGrad.resize(numWorking);
        for(int i = 0, j = 0; i < working.size(); ++i) {
            if(working(i) == 1.0F) {
                workingGrad(j++) = grad(i);
            }
        }

        double gn = workingGrad.norm();
        //printf("\r%5d %5d %15.5e %15.5e %15.5e %15.5e\n",
        //  count, iter, t, fx, workingGrad.cwiseAbs().sum(), gn );

        // Check Optimality
        if(numWorking == 0) {
            RGM_LOG(normal,
                    "[LBFGS Optimization]  All variables are at their bound and no further progress is possible");
            break;
        } else if(gn <= optTol_) {
            RGM_LOG(normal,
                    "[LBFGS Optimization]  All working variables satisfy optimality condition");
            break;
        }

        // Check for lack of progress
        if((t * d).cwiseAbs().sum() < optTol_) {
            RGM_LOG(normal, "[LBFGS Optimization] Step size below optTol");
            break;
        }

        if(std::abs(fx - fx_old) < optTol_) {
            RGM_LOG(normal, "[LBFGS Optimization] Function value changing by less than optTol");
            break;
        }

        if(iter > maxIterations_) {
            break;
        }

        count++;
    } // while

    return fx;
}

} // namespace RGM
