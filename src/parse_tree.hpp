#ifndef RGM_PARSETREE_HPP_
#define RGM_PARSETREE_HPP_

#include "parameters.hpp"

namespace RGM {

/// Predeclaration
template<int Dimension>
class AOGrammar_;

template<int Dimension>
class Node_;

/// The PtEdge represents an instance of an Edge in a grammar
class PtEdge {
  public:
    /// Index
    enum {
        IDX_MYSELF = 0, IDX_FROM, IDX_TO, IDX_G, IDX_TYPE, IDX_NUM
    };

    /// Type of index
    typedef Eigen::Matrix<int, 1, IDX_NUM, Eigen::RowMajor> Index;

    /// Default constructor
    PtEdge() { idx_.fill(-1); }

    /// Destructor
    ~PtEdge() {}

    /// Copy constructor
    explicit PtEdge(const PtEdge & e) { idx_ = e.idx(); }

    /// Constructs an Edge with given type @p fromNode and @p toNode
    explicit PtEdge(int fromNode, int toNode) {
        idx_ << -1, fromNode, toNode, -1, -1;
    }

    /// Constructs an Edge with given type @p fromNode and @p toNode
    explicit PtEdge(int fromNode, int toNode, int gEdge, int type) {
        idx_ << -1, fromNode, toNode, gEdge, type;
    }

    /// Returns the set of indice
    const Index & idx() const { return idx_; }
    Index & getIdx() { return idx_; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
    Index idx_;

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;

}; // class PtEdge


/// The PtNode represents an instance of a Node in a grammar
class PtNode {
  public:
    /// Index
    enum {
        IDX_MYSELF = 0, IDX_TYPE, IDX_BIAS, IDX_DEF, IDX_SCALEPRIOR, IDX_APP,
        IDX_G, IDX_PARSEINFO, IDX_VALID, IDX_NUM
    };

    /// Type of index
    typedef Eigen::Matrix<int, 1, IDX_NUM, Eigen::RowMajor> Index;

    /// Default constructor
    PtNode() { idx_.fill(-1);  idx_(IDX_VALID) = 1; }

    /// Destructor
    ~PtNode() {}

    /// Copy constructor
    PtNode(const PtNode & n);

    /// Constructs a PtNode with given index @p gNode of a Node in a grammar
    PtNode(int gNode);

    /// Returns the indice of in-edges
    const vector<int> & idxInEdges() const { return idxInEdges_; }
    vector<int> & getIdxInEdges() { return idxInEdges_; }

    /// Returns the indice of out-edges
    const vector<int> & idxOutEdges() const { return idxOutEdges_; }
    vector<int> & getIdxOutEdges() { return idxOutEdges_; }

    /// Returns the set of indice
    const Index & idx() const { return idx_; }
    Index & getIdx() { return idx_; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
    vector<int> idxInEdges_;
    vector<int> idxOutEdges_;

    Index idx_;

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;

}; // class PtNode


/// States of a parse tree for learning using WL-SSVM
struct PtStates {
    bool isBelief_;
    Scalar score_;
    Scalar loss_;
    Scalar margin_;
    Scalar norm_;

    /// Default constructor
    PtStates() :
        isBelief_(false), score_(0), loss_(1), margin_(0), norm_(0) {}

    /// Copy constructor
    explicit PtStates(const PtStates & s) :
        isBelief_(s.isBelief_), score_(s.score_), loss_(s.loss_),
        margin_(s.margin_), norm_(s.norm_) {}

    /// Constructs a state with inputs
    explicit PtStates(bool isBelief, Scalar loss, Scalar norm) :
        isBelief_(isBelief), score_(0), loss_(loss), margin_(0), norm_(norm) {}

    /// Constructs a state with inputs
    explicit PtStates(bool isBelief, Scalar score, Scalar loss, Scalar norm) :
        isBelief_(isBelief), score_(score), loss_(loss), margin_(0), norm_(norm) {}

    /// init
    void init();

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;

}; // struct States

/// A ParseTree is a instantiation of a AOGrammar
/// It is defined separately for the simplicity of the data structure
template<int Dimension>
class ParseTree_ {
  public:
    /// Type of a detection
    typedef Detection_<Scalar> Detection;
    typedef typename Appearance::Param AppParam;

    /// Default constructor
    ParseTree_() :
        g_(NULL), idxRootNode_(-1), states_(NULL), dataId_(-1),
        appearanceX_(NULL), imgWd_(0), imgHt_(0) {}

    /// Copy constructor
    // @note The grammar g_ can not be copied from @p pt
    /// which will be set outside constructor
    ParseTree_(const ParseTree & pt);

    /// Destructor
    ~ParseTree_() { clear(); }

    /// Constructs a parse tree for a grammar @p g
    ParseTree_(const AOGrammar & g) :
        g_(&g), idxRootNode_(-1), states_(NULL) {}

    /// Set the grammar for it
    void setGrammar(const AOGrammar & g) { g_ = &g; }

    /// Assign operator
    ParseTree & operator=(const ParseTree & pt);

    /// Compares the scores in decreasing order
    bool operator<(const ParseTree & pt) const;

    /// Swap
    void swap(ParseTree & pt);

    /// Clears
    void clear();

    /// Deletes the subtree rooted at a node
    void deleteSubtree(PtNode * node);

    /// Returns if a parse tree is empty
    bool empty() const { return nodeSet().size() == 0; }

    /// Creates a sample
    void createSample(FeaturePyr & pyr, InferenceParam & param);

    /// Returns the nodeSet
    const vector<PtNode *> & nodeSet() const { return nodeSet_; }
    vector<PtNode *> & getNodeSet() { return nodeSet_; }

    /// Returns the edgeSet
    const vector<PtEdge *> & edgeSet() const { return edgeSet_; }
    vector<PtEdge *> & getEdgeSet() { return edgeSet_; }

    /// Returns the root node
    int            idxRootNode() const { return idxRootNode_; }
    int      &     getIdxRootNode() { return idxRootNode_; }
    const PtNode * rootNode() const { return nodeSet()[idxRootNode()]; }
    PtNode *& getRootNode() { return getNodeSet()[idxRootNode()]; }

    /// Returns the grammar
    const AOGrammar  * grammar() const { return g_; }

    /// Returns the appearance set
    const vector<AppParam *> & appearanceSet() const { return appearanceSet_; }
    vector<AppParam *> & getAppearanceSet() { return appearanceSet_; }

    /// Returns the bias set
    const vector<Scalar>  & biasSet() const { return biasSet_; }
    vector<Scalar> & getBiasSet() { return biasSet_; }

    /// Returns the deformation set
    const vector<Deformation::Param *> & deformationSet() const {
        return deformationSet_;
    }
    vector<Deformation::Param *> & getDeformationSet() { return deformationSet_; }

    /// Returns the Scaleprior set
    const vector<Scaleprior::Param *> & scalepriorSet() const { return scalepriorSet_; }
    vector<Scaleprior::Param *> & getScalepriorSet() { return scalepriorSet_; }

    /// Returns the parse info set
    const vector<ParseInfo *> & parseInfoSet() const { return parseInfoSet_; }
    vector<ParseInfo *> & getParseInfoSet() { return parseInfoSet_; }

    /// Returns the states
    const PtStates  * states() const { return states_; }
    PtStates *& getStates() { return states_; }

    /// Returns dataId
    int  dataId() const { return dataId_; }
    int & getDataId() { return dataId_; }

    int ptId() const { return ptId_; }
    int & getPtId() { return ptId_; }

    /// Returns the appearance usage
    const vector<int> & appUsage() const { return appUsage_; }
    vector<int> & getAppUsage() { return appUsage_; }

    /// Returns the appearance x
    const AppParam  * appearaceX() const { return appearanceX_; }
    AppParam *& getAppearaceX() { return appearanceX_; }

    /// Returns image wd and ht
    int imgWd() const { return imgWd_; }
    int & getImgWd() { return imgWd_; }

    int imgHt() const { return imgHt_; }
    int & getImgHt() { return imgHt_; }

    /// Returns the index of object component
    int idxObjComp() const;

    /// Adds a node
    int addNode(int gNode, int type);

    /// Adds a edge
    int addEdge(int fromNode, int toNode, int gEdge, int type);

    /// Adds a bias
    int AddBias(Scalar w);

    /// Adds a scale prior
    int addScaleprior(Scaleprior::Param & w);

    /// Adds a deformation
    int addDeformation(Scalar dx, Scalar dy, bool flip);

    /// Adds an appearance
    int addAppearance(AppParam & w, featureType t, bool flip);

    /// Adds a parse info
    int addParseInfo(ParseInfo & info);

    /// Visualizes it
    void showDetection(Mat img, bool display = false,
                       cv::Scalar color = cv::Scalar(0, 0, 255),
                       bool showPart = true);

    /// Visualizes
    void visualize(string & saveName, std::map<int, Scalar> * label = NULL);

    /// Returns the length of total concatenated features
    int dim() const;

    /// Compares feature values with another pt
    int compareFeatures(const ParseTree & pt) const;

    /// Computes the norm
    Scalar norm() const;

    /// Computer overlap with a given bbox
    // @param[in] ref The reference box at the original image resolution
    Scalar computeOverlapLoss(const Rectangle & ref) const;

    /// Finds the pt node which corresponds to the specified grammar node
    /// or its idx
    vector<const PtNode *> findNode(const Node_<Dimension> * n);
    vector<const PtNode *> findNode(const int idxG);
    vector<PtNode *> getNode(const int idxG);

    /// Finds the single obj And-nodes
    /// assume the grammar model is one-layer part-based model
    vector<const PtNode *> findSingleObjAndNodes() const;
    vector<PtNode *> getSingleObjAndNodes();

    /// Gets single object detections
    void getSingleObjDet(vector<Detection> & dets, int ptIdx = -1);

    /// Do the bbox prediction
    void doBboxPred(vector<Detection> & dets, int ptIdx = -1);

    /// Gets score
    Scalar score();

    /// Gets root node parse info
    const ParseInfo * rootParseInfo() const;

    /// Get bounding box
    Rectangle bbox();

    /// Returns the fromNode
    const PtNode * fromNode(const PtEdge & e) const;
    PtNode * getFromNode(const PtEdge & e);
    const PtNode * fromNode(const PtEdge * e) const;
    PtNode * getFromNode(const PtEdge * e);

    /// Returns the fromNode
    const PtNode * toNode(const PtEdge & e) const;
    PtNode * getToNode(const PtEdge & e);
    const PtNode * toNode(const PtEdge * e) const;
    PtNode * getToNode(const PtEdge * e);

    /// Returns an InEdge with idxInEdge_[i]
    const PtEdge * inEdge(int i, const PtNode & n) const;
    PtEdge * getInEdge(int i, const PtNode & n);
    const PtEdge * inEdge(int i, const PtNode * n) const;
    PtEdge * getInEdge(int i, const PtNode * n);

    /// Returns an OutEdge with idxOutEdge_[i]
    const PtEdge * outEdge(int i, const PtNode & n) const;
    PtEdge * getOutEdge(int i, const PtNode & n);
    const PtEdge * outEdge(int i, const PtNode * n) const;
    PtEdge * getOutEdge(int i, const PtNode * n);

    /// Returns the parse info
    const ParseInfo  * parseInfo(const PtNode & n) const;
    ParseInfo *& getParseInfo(const PtNode & n);
    const ParseInfo  * parseInfo(const PtNode * n) const;
    ParseInfo *& getParseInfo(const PtNode * n);

  private:
    vector<PtNode *>  nodeSet_;
    vector<PtEdge *>  edgeSet_;
    int idxRootNode_;

    const AOGrammar * g_;

    vector<AppParam *> appearanceSet_;
    vector<Scalar>  biasSet_;
    vector<Deformation::Param *> deformationSet_;
    vector<Scaleprior::Param *> scalepriorSet_;

    vector<ParseInfo *> parseInfoSet_;

    int dataId_;
    PtStates * states_;

    int ptId_;

    vector<int> appUsage_;

    // for root 2x used in searching part configuration
    AppParam * appearanceX_;

    int imgWd_;
    int imgHt_;

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;

}; // class ParseTree


/// A train sample is a set of parse trees
template<int Dimension>
class TrainSample_ {
  public:
    /// Type of parse tree iterator
    typedef typename vector<ParseTree >::iterator  ptIterator;

    /// Constants
    static const int NbHist = 50;

    /// Default constructor
    TrainSample_() :
        marginBound_(0), beliefNorm_(0), maxNonbeliefNorm_(0), nbHist_(0) {}

    /// Copy constructor
    TrainSample_(const TrainSample & ex);

    /// Assignment operator
    TrainSample & operator=(const TrainSample & ex);

    void swap(TrainSample & ex);

    /// Returns parse trees
    const vector<ParseTree *> & pts() const { return pts_; }
    vector<ParseTree *> & getPts() { return pts_; }

    /// Returns the margin bound
    Scalar   marginBound() const { return marginBound_; }
    Scalar & getMarginBound() { return marginBound_; }

    /// Returns norms of belief and non-belief parse trees
    Scalar   beliefNorm() const { return beliefNorm_; }
    Scalar & getBeliefNorm() { return beliefNorm_; }
    Scalar   maxNonbeliefNorm() const { return maxNonbeliefNorm_; }
    Scalar & getMaxNonbeliefNorm() { return maxNonbeliefNorm_; }

    /// Returns the number of historical record
    int   nbHist() const { return nbHist_; }
    int & getNbHist() { return nbHist_; }

    /// Checks the duplication with a parse tree
    bool isEqual(const ParseTree & pt) const;

  private:
    vector<ParseTree *> pts_;

    // For keeping track on the bound that determines if a an
    // example might possibly have a non-zero loss
    Scalar marginBound_;

    // Maximum L2 norm of the feature vectors for this example
    // (used in conjunction with margin_bound)
    Scalar beliefNorm_;
    Scalar maxNonbeliefNorm_;

    int nbHist_;

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;
};


/// A set of train samples
template<int Dimension>
class TrainSampleSet_ : public vector<TrainSample > {
  public:
    /// Constructor
    TrainSampleSet_() :
        vector<TrainSample_<Dimension> >() {}

    /// Computes Loss
    Scalar computeLoss(bool isPos, Scalar C, int start = 0);

    /// Get all pts
    vector<ParseTree *> getAllNonZeroPts();

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;
};



/// Functor used to test for the intersection of two parse trees
/// according to the Pascal criterion (area of intersection over area of union).
template<int Dimension>
class PtIntersector_ {
  public:
    /// Constructor.
    /// @param[in] reference The reference parse tree.
    /// @param[in] threshold The threshold of the criterion.
    /// @param[in] dividedByUnion Use Felzenszwalb's criterion instead
    /// (area of intersection over area of second rectangle).
    /// Useful to remove small detections inside bigger ones.
    PtIntersector_(const ParseTree_<Dimension> & reference,
                   Scalar threshold = 0.5, bool dividedByUnion = false);

    /// Tests for the intersection between a given rectangle and the reference.
    /// @param[in] rect The rectangle to intersect with the reference.
    /// @param[out] score The score of the intersection.
    bool operator()(const ParseTree_<Dimension> & pt, Scalar * score = 0) const;

  private:
    const ParseTree_<Dimension> * reference_;
    Scalar threshold_;
    bool   dividedByUnion_;

    DEFINE_RGM_LOGGER;
};

} // namespace RGM

#endif // RGM_PARSETREE_HPP_


