#ifndef RGM_PARAMETERS_HPP_
#define RGM_PARAMETERS_HPP_

#include "feature_pyr.hpp"

namespace RGM {

/// The ParamUtil class consists of the common utility parameters
/// used in learning the grammar
class ParamUtil {
  public:
    /// Default constructor
    ParamUtil() : learningStatus_(0), regularizationCost_(0) {}

    /// Default destructor
    virtual ~ParamUtil() {}

    /// Returns the learning status of the offset parameter
    Scalar   learningStatus() const { return learningStatus_; }
    Scalar & getLearningStatus() { return learningStatus_; }

    /// Returns the regularization cost of the offset parameter
    Scalar   regularizationCost() const { return regularizationCost_; }
    Scalar & getRegularizationCost() { return regularizationCost_; }

  protected:
    // 0: parameters are not learned; >0: they are learned
    Scalar learningStatus_;
    Scalar regularizationCost_;

    DEFINE_SERIALIZATION;
};


/// The Appearance class represents the appearance parameters of a TERMINAL-node
///  in the grammar or the features in the feature pyramid of a deformed
///  TERMINAL-node instantiated in the parse tree
template<int Dimension>
class Appearance_ : public ParamUtil {

  public:
    /// Type of parameter vector
    typedef typename FeaturePyr::Level  Param;
    typedef typename FeaturePyr::dLevel dParam;
    /// Type of cell
    typedef typename FeaturePyr::Cell  Cell;
    typedef typename FeaturePyr::dCell dCell;

    /// Default constuctor
    Appearance_() : type_(UNSPECIFIED_FEATURE) {}

    explicit Appearance_(featureType t) : type_(t) {
        RGM_CHECK_EQ(FeatureDim[static_cast<int>(type())], Dimension);
    }

    /// Copy constructor
    Appearance_(const Appearance & app);

    /// Default destructor
    virtual ~Appearance_() {}

    /// Init
    void init(int wd, int ht);

    /// Reset
    void reset();

    /// Returns type
    featureType type() const { return type_; }
    featureType & getType() { return type_; }

    /// Returns the param
    const Param & w() const { return w_; }
    Param & getW() { return w_; }

    /// Returns the lower bound
    const Param & lowerBound() const { return lowerBound_; }
    Param & getLowerBound() { return lowerBound_; }

    /// Returns the gradient
    const dParam & gradient() const { return gradient_; }
    dParam & getGradient() { return gradient_; }

  private:
    featureType type_;
    Param  w_;
    Param  lowerBound_;
    dParam  gradient_; // gradient used in learning

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;

}; // class Appearance


/// The Bias class represents the bias term
class Bias : public ParamUtil {
  public:
    /// Default constructor
    Bias() : w_(0), lowerBound_(0), gradient_(0) {}

    /// Copy constructor
    explicit Bias(const Bias & b);

    /// Default destructor
    virtual ~Bias() {}

    /// Returns the offset parameter
    Scalar   w() const { return w_; }
    Scalar & getW() { return w_; }

    /// Returns the parameter lower bound
    Scalar   lowerBound() const { return lowerBound_; }
    Scalar & getLowerBound() { return lowerBound_; }

    /// Returns the gradient
    const double & gradient() const { return gradient_; }
    double & getGradient() { return gradient_;}

    /// Resets
    void reset();

  private:
    Scalar  w_;
    Scalar  lowerBound_;

    double  gradient_;

    DEFINE_SERIALIZATION;

}; // class Offset



/// The ScalePrior class represents the scale prior parameter
class Scaleprior : public ParamUtil {
  public:
    /// Type of parameter vector
    typedef Eigen::Matrix<Scalar, 1, 3, RowMajor> Param;

    /// Type of parameter gradient
#ifndef RGM_USE_DOUBLE
    typedef Eigen::Matrix<double, 1, 3, RowMajor> dParam;
#else
    typedef Param  dParam;
#endif

    /// Type of a row vector
    typedef Eigen::Matrix<Scalar, 1, Dynamic, RowMajor>  Vector;

    /// Default constructor
    Scaleprior() : w_(Param::Zero()), lowerBound_(Param::Zero()) {}

    /// Copy constructor
    explicit Scaleprior(const Scaleprior & prior);

    /// Default destructor
    virtual ~Scaleprior() {}

    /// Returns the scale prior parameter
    const Param &  w() const { return w_; }
    Param &  getW() { return w_; }

    /// Returns the parameter lower bound
    const Param & lowerBound() const { return lowerBound_; }
    Param & getLowerBound() { return lowerBound_; }

    /// Returns the gradient
    const dParam & gradient() const { return gradient_; }
    dParam & getGradient() { return gradient_; }

    /// Resets
    void reset(bool allow);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  private:
    Param  w_;
    Param  lowerBound_;

    dParam  gradient_;

    DEFINE_SERIALIZATION;

}; // class Scaleprior



/// The Deformation class represents a 2d quadratic deformation (dx^2 dx dy^2 dy).
class Deformation : public ParamUtil {
  public:
    /// Type of a 2d quadratic deformation
    typedef Eigen::Matrix<Scalar, 4, 1, Eigen::ColMajor, 4, 1> Param;

#ifndef RGM_USE_DOUBLE
    typedef Eigen::Matrix<double, 4, 1, Eigen::ColMajor, 4, 1> dParam;
#else
    typedef Param   dParam;
#endif

    /// Bounded shift in DT
    static const int BoundedShiftInDT = 4;

    /// Default constructor
    Deformation() : w_(Param::Zero()), lowerBound_(Param::Zero()) {}

    /// Copy constructor
    explicit Deformation(const Deformation & def);

    /// Constructs a deformation with given arguments
    explicit Deformation(Scalar dx, Scalar dy);

    /// Default destructor
    virtual ~Deformation() {}

    /// Returns the scale prior parameter
    const Param &  w() const { return w_; }
    Param &  getW() { return w_; }

    /// Returns the parameter lower bound
    const Param & lowerBound() const { return lowerBound_; }
    Param & getLowerBound() { return lowerBound_; }

    /// Returns the gradient
    const dParam & gradient() const { return gradient_; }
    dParam & getGradient() { return gradient_; }

    /// Resets: 0-for category model, 1-for specific tracker
    void reset(int t=0);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
    Param  w_;
    Param  lowerBound_;

    dParam  gradient_;

    DEFINE_SERIALIZATION;

}; // class Deformation


} // namespace RGM


#endif // RGM_PARAMETERS_HPP_




