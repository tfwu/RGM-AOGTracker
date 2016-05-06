#include "parameters.hpp"
#include "util/UtilSerialization.hpp"

namespace RGM {

// ------ ParamUtil ------

template<class Archive>
void ParamUtil::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(learningStatus_);
    ar & BOOST_SERIALIZATION_NVP(regularizationCost_);
}

INSTANTIATE_BOOST_SERIALIZATION(ParamUtil);



// ------ Appearance ------

template<int Dimension>
Appearance::Appearance_(const Appearance & app) {
    getType()               = app.type();
    getW()                  = app.w();
    getLowerBound()         = app.lowerBound();
    getLearningStatus()     = app.learningStatus();
    getRegularizationCost() = app.regularizationCost();
}

template<int Dimension>
void Appearance::init(int wd, int ht) {
    getW()                  = Param::Constant(ht, wd, Cell::Zero());
    getLowerBound()         = Param::Constant(ht, wd, Cell::Constant(-Inf));

    getRegularizationCost() = 1.0F;
    getLearningStatus()     = 1.0F;
}

template<int Dimension>
void Appearance::reset() {
    getW().setConstant(Cell::Zero());
    getLowerBound().setConstant(Cell::Constant(-Inf));

    getRegularizationCost() = 1.0F;
    getLearningStatus()     = 1.0F;
}

template<int Dimension>
template<class Archive>
void Appearance::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ParamUtil);
    ar & BOOST_SERIALIZATION_NVP(type_);
    ar & BOOST_SERIALIZATION_NVP(w_);
    ar & BOOST_SERIALIZATION_NVP(lowerBound_);
}

INSTANTIATE_CLASS_(Appearance_);
INSTANTIATE_BOOST_SERIALIZATION_(Appearance_);


// ------ Bias ------

Bias::Bias(const Bias & b) {
    getW()                  = b.w();
    getLowerBound()         = b.lowerBound();
    getLearningStatus()     = b.learningStatus();
    getRegularizationCost() = b.regularizationCost();
}

void Bias::reset() {
    getW() = 0;
    getLowerBound() = -Inf;
    getRegularizationCost() = 0;
    getLearningStatus() = 20;
}

template<class Archive>
void Bias::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ParamUtil);
    ar & BOOST_SERIALIZATION_NVP(w_);
    ar & BOOST_SERIALIZATION_NVP(lowerBound_);
}

INSTANTIATE_BOOST_SERIALIZATION(Bias);



// ------ Scaleprior ------

Scaleprior::Scaleprior(const Scaleprior & prior) {
    getW()                  = prior.w();
    getLowerBound()         = prior.lowerBound();
    getLearningStatus()     = prior.learningStatus();
    getRegularizationCost() = prior.regularizationCost();
}

void Scaleprior::reset(bool allow) {
    if(allow) {
        getW().setZero();
        getLowerBound() = Scaleprior::Param::Constant(-Inf);
        getRegularizationCost() = 1;
        getLearningStatus() = 1;
    } else {
        // [-1000 0 0] prevents the root filter from being placed
        // in the bottom octave of the feature pyramid
        getW() << -1000, 0, 0;
        getLowerBound() = Scaleprior::Param::Constant(-Inf);
        getRegularizationCost() = 0;
        getLearningStatus() = 0;
    }
}

template<class Archive>
void Scaleprior::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ParamUtil);
    ar & BOOST_SERIALIZATION_NVP(w_);
    ar & BOOST_SERIALIZATION_NVP(lowerBound_);
}

INSTANTIATE_BOOST_SERIALIZATION(Scaleprior);



// ------ Deformation ------

Deformation::Deformation(const Deformation & def) {
    getW()                  = def.w();
    getLowerBound()         = def.lowerBound();
    getLearningStatus()     = def.learningStatus();
    getRegularizationCost() = def.regularizationCost();
}

Deformation::Deformation(Scalar dx, Scalar dy) :
    lowerBound_(Param::Zero()) {
    getW() << dx * dx, dx, dy * dy, dy;
}

void Deformation::reset(int t) {
    if ( t == 0 ) {
        getW() << 0.01F, 0.0F, 0.01F, 0.0F;
        getLowerBound() << 0.001F, -Inf,  0.001F,  -Inf;
    } else {
        getW() << 0.05F, 0.0F, 0.05F, 0.0F;
        getLowerBound() << 0.01F, -Inf,  0.01F,  -Inf;
    }
    getLearningStatus() = 0.1;
    getRegularizationCost() = 10;
}

template<class Archive>
void Deformation::serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(ParamUtil);
    ar & BOOST_SERIALIZATION_NVP(w_);
    ar & BOOST_SERIALIZATION_NVP(lowerBound_);
}

INSTANTIATE_BOOST_SERIALIZATION(Deformation);

} // namespace RGM
