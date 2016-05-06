#ifndef RGM_AOGRAMMAR_HPP_
#define RGM_AOGRAMMAR_HPP_

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "parameters.hpp"
#include "patch_work.hpp"
#include "AOGrid.hpp"
#include "util/UtilSerialization.hpp"

namespace RGM {

/// Predeclaration
template<int Dimension>
class Node_;

template<int Dimension>
class Edge_;

template<int Dimension>
class AOGrammar_;

template<int Dimension>
class ParseTree_;


/// The Edge class represents a Rule
/// (switching, composition, deformation or termination) in the grammar
template<int Dimension>
class Edge_ {

  public:
    /// Index
    enum {
        IDX_FROM = 0, IDX_TO, IDX_MIRROR, IDX_NUM
    };

    typedef Eigen::Matrix<int, 1, IDX_NUM>  Index;

    /// Constructs an empty Edge
    Edge_() { init(); }

    /// Copy constructor
    explicit Edge_(const Edge & e);

    /// Constructs an Edge with given type @p t, @p fromNode and @p toNode
    explicit Edge_(edgeType t, Node * fromNode, Node * toNode);

    /// Destructor
    ~Edge_() {}

    /// Returns the edge type
    edgeType   type() const { return edgeType_; }
    edgeType & getType() { return edgeType_; }

    /// Returns the start Node
    const Node  * fromNode() const { return fromNode_; }
    Node *& getFromNode() { return fromNode_; }

    /// Returns the end Node
    const Node  * toNode() const { return toNode_; }
    Node *& getToNode() { return toNode_; }

    /// Returns if it is mirrored Node
    bool   isLRFlip() const { return isLRFlip_; }
    bool & getIsLRFlip() { return isLRFlip_; }

    /// Return left-right mirrored node
    const Edge  * lrMirrorEdge() const { return LRMirrorEdge_; }
    Edge *& getLRMirrorEdge() { return LRMirrorEdge_; }

    /// Returns the onoff status
    bool onOff() const { return onOff_; }
    bool & getOnOff() { return onOff_; }

    /// Returns the index
    const Index & idx() const { return idx_; }
    Index & getIdx() { return idx_; }

    /// Assigns the index
    void assignIdx(AOGrammar * g);

    /// Assigns the pointers to set up connections
    void assignConnections(AOGrammar * g);

  private:
    void init();

  private:
    edgeType edgeType_;
    Node * fromNode_;
    Node * toNode_;

    bool isLRFlip_;
    Edge * LRMirrorEdge_;

    bool onOff_;

    Index idx_;

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;

}; /// class Edge



/// The Node class represents a Symbol
/// (non-terminal or terminal) in the grammar
template<int Dimension>
class Node_ {

  public:
    /// Index
    enum {
        IDX_MIRROR = 0, IDX_SCALEPRIOR, IDX_BIAS, IDX_DEF, IDX_APP, IDX_FILTER,
        IDX_AOGRID, IDX_NUM
    };

    typedef Eigen::Matrix<int, 1, IDX_NUM> Index;

    /// Type of FFT filter
    typedef typename Patchwork::Filter Filter;    
    /// Type of appearance parameter
    typedef typename Appearance::Param AppParam;

    /// Default constructor
    Node_() { init(); }

    /// Constructs an empty node of specific type
    explicit Node_(nodeType t);

    /// Copy constructor
    explicit Node_(const Node & n);

    /// Destructor
    ///@note All memory management are done in the AOGrammar
    ~Node_() {}

    /// Returns node type
    nodeType   type() const { return nodeType_; }
    nodeType & getType() { return nodeType_; }

    /// Returns in-edges
    const vector<Edge *> & inEdges() const { return inEdges_; }
    vector<Edge *> & getInEdges() { return inEdges_; }

    /// Returns out-edges
    const vector<Edge *> & outEdges() const { return outEdges_; }
    vector<Edge *> & getOutEdges() { return outEdges_; }

    /// Returns if it is mirrored Node
    bool   isLRFlip() const { return isLRFlip_; }
    bool & getIsLRFlip() { return isLRFlip_; }

    /// Return left-right mirrored node
    const Node  * lrMirrorNode() const { return LRMirrorNode_; }
    Node *& getLRMirrorNode() { return LRMirrorNode_; }

    /// Returns the detection window
    const Rectangle & detectWindow() const { return detectWindow_; }
    Rectangle & getDetectWindow() { return detectWindow_; }

    /// Returns Anchor
    const Anchor & anchor() const { return anchor_; }
    Anchor & getAnchor() { return anchor_; }

    /// Returns the scale prior feature
    const Scaleprior  * scaleprior() const { return scaleprior_; }
    Scaleprior *& getScaleprior() { return scaleprior_; }

    /// Returns the offset
    const Bias  * bias() const { return bias_; }
    Bias *& getBias() { return bias_; }

    /// Returns deformation
    const Deformation  * deformation() const { return deformation_; }
    Deformation *& getDeformation() { return deformation_; }

    /// Returns deformation parameters with proper flipping
    Deformation::Param deformationParam() const;

    /// Returns appearance
    const Appearance  * appearance() const { return appearance_; }
    Appearance *& getAppearance() { return appearance_; }

    /// Returns appearance parameters with proper flipping
    AppParam appearanceParam() const;

    /// Returns the FFT fiter
    const Filter  * cachedFilter() const { return cachedFilter_; }
    Filter *& getCachedFilter() { return cachedFilter_; }

    /// Returns the tag
    const boost::uuids::uuid & tag() const { return tag_; }

    /// Returns the idx
    const vector<int> & idxInEdge() const { return idxInEdge_; }
    vector<int> & getIdxInEdge() { return idxInEdge_; }

    const vector<int> & idxOutEdge() const { return idxOutEdge_; }
    vector<int> & getIdxOutEdge() { return idxOutEdge_; }

    /// Returns the on off status
    bool onOff() const { return onOff_; }
    bool & getOnOff() { return onOff_; }

    const Index  & idx() const { return idx_; }
    Index  & getIdx() { return idx_; }

    /// Assigns the index
    void assignIdx(AOGrammar * g);

    /// Assigns the connections to the pointers
    /// @param[in] g The grammar to which it belongs to
    void assignConnections(AOGrammar * g);

  private:
    /// Init
    void init();

  private:
    nodeType nodeType_;
    vector<Edge *>  inEdges_;
    vector<Edge *>  outEdges_;

    bool isLRFlip_;
    Node  * LRMirrorNode_;

    Rectangle detectWindow_; // size
    Anchor anchor_; // relative (location, scale) w.r.t. the parent node

    /// Pointers to parameters
    Scaleprior  * scaleprior_;
    Bias    *   bias_;
    Deformation * deformation_;

    Appearance * appearance_; // for T-nodes
    Filter * cachedFilter_;

    boost::uuids::uuid tag_; // used to index score maps in inference

    bool onOff_;

    /// utility members used in save and read
    vector<int> idxInEdge_;
    vector<int> idxOutEdge_;

    Index idx_;

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;

}; /// class Node


/// AND-OR Grammar is embedded into an directed and acyclic AND-OR Graph
template<int Dimension>
class AOGrammar_ {

  public:
    /// Type of a detection
    typedef Detection_<Scalar> Detection;
    /// Type of FFT filter
    typedef typename Node::Filter Filter;    
    /// Type of appearance parameter
    typedef typename Appearance::Param AppParam;
    /// Type of feature cell
    typedef typename FeaturePyr::Cell Cell;
    /// Type of feature cell
    typedef typename FeaturePyr::dCell dCell;

    /// Default constructor
    AOGrammar_() { init(); }

    /// Destructor
    ~AOGrammar_() { clear(); }

    /// Constructs a grammar from a saved file
    explicit AOGrammar_(const string & modelFile);

    /// Init root templates
    void initRoots(const vector<pair<int, int> > & rootSz);

    /// Pursues the model structure of DPM by Prof. Felzenszwalb
    void createDPM(int numParts, const pair<int, int> & partSz, int partScale);

    /// Updates RGM
    void createRGM(int partScale, bool useScalePrior=true);

    /// Add RGM for a subcategory
    Node * createRGM(AOGrid * grid, AppParam & w, int partScale);

    /// Turns on a subcategory root model only
    void turnOnRootOnly(const vector<int> & idx);

    /// Init
    void init();

    /// Clears the grammar
    void clear();

    /// Returns if a grammar is empty
    bool empty() const { return nodeSet().size() == 0; }

    /// Returns grammar parameters
    const GrammarParam & gParam() const { return gParam_; }
    GrammarParam & getGParam() { return gParam_; }
    /// Returns type
    grammarType type() const { return gParam_.gType_; }
    grammarType  & getType() { return gParam_.gType_; }
    /// Returns name
    const string & name() const { return gParam_.name_; }
    string & getName() { return gParam_.name_; }
    /// Returns note
    const string & note() const { return gParam_.note_; }
    string & getNote() { return gParam_.note_; }
    /// Returns year
    const string & year() const { return gParam_.year_; }
    string & getYear() { return gParam_.year_; }
    /// Returns if the model has not been trained yet
    bool isZero() const { return gParam_.isZero_; }
    bool & getIsZero() { return gParam_.isZero_; }
    /// Returns if the model is for specific object (e.g. in tracking)
    bool isSpecific() const { return gParam_.isSpecific_; }
    bool & getIsSpecific() { return gParam_.isSpecific_; }
    /// Returns isSingleObjModel
    bool isSingleObjModel() const { return gParam_.isSingleObjModel_; }
    bool & getIsSingleObjModel() { return gParam_.isSingleObjModel_; }
    /// Returns the status of having lr flip components
    bool isLRFlip() const { return gParam_.isLRFlip_; }
    bool & getIsLRFlip()  { return gParam_.isLRFlip_; }
    bool sharedLRFlip() const { return gParam_.sharedLRFlip_; }
    bool & getSharedLRFlip() { return gParam_.sharedLRFlip_; }
    /// Returns the regularization type
    regType regMethod() const { return gParam_.regMethod_; }
    regType & getRegMethod() { return gParam_.regMethod_; }

    /// Returns the feature extraction param.
    const FeatureParam & featParam() const { return featureParam_; }
    FeatureParam & getFeatParam() { return featureParam_; }
    /// Returns feature type
    featureType featType() const { return featureParam_.type_; }
    featureType & getFeatType() { return featureParam_.type_; }
    /// Returns the cell size used to extract features
    int cellSize() const { return featureParam_.cellSize_; }
    int & getCellSize() { return featureParam_.cellSize_; }
    int minCellSize() {
        return extraOctave() ? cellSize() / 4 : cellSize() / 2;
    }
    /// Returns if extra-octave is used in feature pyramid
    bool extraOctave() const { return featureParam_.extraOctave_; }
    bool & getExtraOctave() { return featureParam_.extraOctave_; }
    /// Returns the feature bias
    Scalar featureBias() const { return featureParam_.featureBias_; }
    Scalar & getFeatureBias() { return featureParam_.featureBias_; }
    /// Returns the interval of pyramid
    int   interval() const { return featureParam_.interval_; }
    int & getInterval() { return featureParam_.interval_; }
    /// Returns the padding
    int padx() const { return featureParam_.padx_; }
    int pady() const { return featureParam_.pady_; }

    /// Returns the bg mu
    const Cell & bgmu() const { return bgmu_; }
    Cell & getBgmu() { return bgmu_; }

    /// Returns the node set
    const vector<Node *> & nodeSet() const { return nodeSet_; }
    vector<Node *> & getNodeSet() { return nodeSet_; }

    /// Return the root node
    const Node  * rootNode() const { return rootNode_; }
    Node *& getRootNode() { return rootNode_; }

    /// Returns the number of subcategories
    int nbSubcategories(bool isOnOnly, bool withLRFlip);

    /// Returns the subcategory root And-Nodes
    vector<Node *> getSubcategoryRootAndNodes(bool isOnOnly, bool withLRFlip);

    /// Traces the DFS ordering of nodes
    void traceNodeDFS(Node * curNode, vector<int> & visited,
                      vector<Node *> & nodeDFS, bool onlyOnNodes = true);
    /// Returns the DFS node set
    const vector<Node *> & nodeDFS() const { return nodeDFS_; }
    vector<Node *> & getNodeDFS() { return nodeDFS_; }
    const vector<vector<Node *> > & compNodeDFS() const { return compNodeDFS_; }
    vector<vector<Node *> > & getCompNodeDFS() { return compNodeDFS_; }

    /// Traces the DFS ordering of nodes
    void traceNodeBFS(Node * curNode, vector<int> & visited,
                      vector<Node *> & nodeBFS, bool onlyOnNodes = true);
    /// Returns the DFS node set
    const vector<Node *> & nodeBFS() const { return nodeBFS_; }
    vector<Node *> & getNodeBFS() { return nodeBFS_; }
    const vector<vector<Node *> > & compNodeBFS() const { return compNodeBFS_; }
    vector<vector<Node *> > & getCompNodeBFS() { return compNodeBFS_; }

    /// Computes DFS and BFS
    void traceDFSandBFS(bool onlyOnNodes = true);
    /// Computes the component-based DFS/BFS
    void traceCompNodeDFSandBFS(bool onlyOnNodes = true);

    /// Returns the edge set
    const vector<Edge *> & edgeSet() const { return edgeSet_; }
    vector<Edge *> & getEdgeSet() { return edgeSet_; }

    /// Returns the appearance set
    const vector<Appearance *>  & appearanceSet() const {
        return appearanceSet_;
    }
    vector<Appearance *>  & getAppearanceSet() { return appearanceSet_; }

    /// Returns the bias set
    const vector<Bias *>  & biasSet() const { return biasSet_; }
    vector<Bias *>  & getBiasSet() { return biasSet_; }

    /// Returns the deformation set
    const vector<Deformation *>  & deformationSet() const { return deformationSet_; }
    vector<Deformation *>  & getDeformationSet() { return deformationSet_; }

    /// Returns the scaleprior set
    const vector<Scaleprior *> & scalepriorSet() const { return scalepriorSet_; }
    vector<Scaleprior *> & getScalepriorSet() { return scalepriorSet_; }

    /// Returns the size of max detection window
    const Rectangle & maxDetectWindow() const { return maxDetectWindow_; }
    Rectangle & getMaxDetectWindow() { return maxDetectWindow_; }

    /// Returns the size of min detection window
    const Rectangle & minDetectWindow() const { return minDetectWindow_; }
    Rectangle & getMinDetectWindow() { return minDetectWindow_; }

    /// Returns the cached FFT filters
    const vector<Filter *>  & cachedFilters() const { return cachedFilters_; }
    vector<Filter *>  & getCachedFilters() { return cachedFilters_; }

    /// Returns the status of caching
    bool   cachedFFTStatus() const { return cached_; }
    bool & getCachedFFTStatus() { return cached_; }

    /// Transfers the filters of T-nodes
    void cachingFilters();

    /// Returns the AOGrids
    const vector<AOGrid *> & gridAOG() const { return gridAOG_; }
    vector<AOGrid *> & getGridAOG() { return gridAOG_; }

    /// Returns the threshold
    Scalar   thresh() const { return thresh_; }
    Scalar & getThresh() { return thresh_; }

    /// Returns the bbox prediction model
    const std::map<int, Matrix> & bboxPred() const { return bboxPred_; }
    std::map<int, Matrix> & getBboxPred() { return bboxPred_; }

    /// Gets the length of total parameters
    int dim() const;

    /// Returns the index of a Node in the set of nodes
    int idxNode(const Node * node) const;
    /// Returns the index of a Node in the children set of its single parent Node
    int idxNodeAsChild(const Node * node) const;
    /// Returns the index of a parent And-node given a input T-node
    int idxParentAndNodeOfTermNode(const Node * tnode) const;
    /// Returns the index of an Edge in the set of edges
    int idxEdge(const Edge * edge) const;
    /// Returns the index of an Appearance in the set of appearance
    int idxAppearance(const Appearance * app) const;
    /// Returns the index of an Offset in the set of offsets
    int idxBias(const Bias * b) const;
    /// Returns the index of a Deformation in the set of deformation
    int idxDeformation(const Deformation * def) const;
    /// Returns the index of a Scaleprior in the set of scaleprior
    int idxScaleprior(const Scaleprior * scale) const;
    /// Returns the index of a fft filter
    int idxFilter(const Filter  * filter) const;

    /// Returns the Node with the given index
    Node * findNode(int idx);
    const Node * findNodeConst(int idx) const;
    /// Returns the Edge with the given index
    Edge * findEdge(int idx);
    /// Returns the Appearance with the given index
    Appearance * findAppearance(int idx);
    /// Returns the Offset with the given index
    Bias * findBias(int idx);
    /// Returns the Deformation with the given index
    Deformation * findDeformation(int idx);
    /// Returns the scaleprior with the given index
    Scaleprior * findScaleprior(int idx);
    /// Returns the fft filter with the given index
    Filter  * findCachedFilter(int idx);

    /// Saves to a stream
    /// @param[in] archiveType 0 binary, 1 text
    void save(const string & modelFile, int archiveType = 0);
    /// Reads from a stream
    bool read(const string & modelFile, int archiveType = 0);

    /// Visualize the grammar using GraphViz
    string visualize(const string & saveDir, string saveName = "",
                     ParseTree * pt = NULL, bool createApp = true,
                     bool onlyOnNodes = false,
                     vector<bool> * nodeOnOffHistory = NULL,
                     vector<bool> * edgeOnOffHistory = NULL,
                     string extraImgFile = "");
    string visualize1(const string & saveDir, string saveName = "",
                     ParseTree * pt = NULL, std::map<int, cv::Mat> *ptScoreMaps=NULL,
                     bool createApp = true,
                     bool onlyOnNodes = false,
                     vector<bool> * nodeOnOffHistory = NULL,
                     vector<bool> * edgeOnOffHistory = NULL);

    void visualize(const string &saveDir, Mat img, string saveName);

    /// Adds a node
    Node * addNode(nodeType t);
    /// Adds an edge
    Edge * addEdge(Node * from, Node * to, edgeType t);
    /// Adds a bias
    Bias * addBias(const Bias & bias);
    /// Adds a scale prior
    Scaleprior * addScaleprior(const Scaleprior & prior);
    /// Adds a deformation
    Deformation * addDeformation(const Deformation & def);
    /// Adds a child node
    pair<Node *, Edge *> addChild(Node * parent, nodeType chType,
                                  edgeType edgeType);
    /// Adds a parent node
    pair<Node *, Edge *> addParent(Node * ch, nodeType paType,
                                   edgeType edgeType);
    /// Adds a mirrored copy of a subgraph rooted at a given node
    Node * addLRMirror(Node * root, bool shareParam);

    /// Links two nodes as a left-right pair
    void setNodeLRFlip(Node * n, Node * nFlip);
    /// Links two edges as a left-right pair
    void setEdgeLRFlip(Edge * e, Edge * eFlip);

    /// Adds an apperance unit: OR-node -> AND-node -> T-node
    Node * addAppearanceUnit(const pair<int, int> & sz,
                             const Anchor & anchor,
                             bool hasBias, bool hasScaleprior,
                             bool hasDef);

    Node * addAppearanceUnit(AppParam & w,
                             const Anchor & anchor,
                             bool hasBias, bool hasScaleprior,
                             bool hasDef);

    /// Init all other appearance based on root template
    /// when the model is initialized based on specificAOG_
    /// @param target  nodes turned off(<0), nodes turned on(>0), all nodes(0)
    void initAppFromRoot(int ds, int target = 0, bool doNormalization = true);

    void initAppFromRoot(int ds, const std::vector<int> &idxNotUpdate,
                         bool doNormalization = false);

    /// Finalizes the grammar
    void finalize(bool hasIdx);

    /// Sets on-off status for all nodes and edges
    void setOnOff(bool on);
    /// Sets on-off based on the selected pts
    void setOnOff(const int gridIdx, const vector<int> &ptBFS);
    /// Gets on-off status of all nodes
    void getOnOff(vector<bool> & statusNodes);
    /// Gets on-off status of all nodes and edges
    void getOnOff(vector<bool> & statusNodes, vector<bool> & statusEdges);
    /// Turns off nodes w.r.t. tnode ids
    void turnOff(vector<int> & turnedOffTnodeIdx);

    /// Returns if the AOG is a DAG
    bool isDAG() const;

    /// Assigns grammar parameters
    void assignParameters(const double * x);
    /// Gets grammar parameters, lb, or gradients
    /// @param[in] which Three opitions:
    ///  0 - parameters, 1 - lower bound, 2 - gradient
    void getParameters(double * x, int which);

    /// Init the gradient vector
    void initGradient();
    /// Update gradient
    void updateGradient(const ParseTree & pt, double mult);

    /// Computes the dot product
    /// @note Assume the correspondence between @p g and this grammar
    double dot(const ParseTree & pt);

    /// Computes the norm
    double computeNorm(bool hasGrad);

    /// Computes the L2 norm
    double computeL2Norm(bool hasGrad);

    /// Computes max norm
    double computeMaxNorm(bool hasGrad);

    /// Computes max memory (in MB) for a pt roughly (less than actual size)
    Scalar computeMaxPtMemory();

  private:
    /// Visualizes appearance templates of T-nodes
    void pictureTNodes(const string & saveDir, bool onlyOnNodes = false,
                       std::map<int, cv::Mat> *ptScoreMaps=NULL);
    /// Visualizes deformation
    void pictureDeformation(const string & saveDir, bool onlyOnNodes = false,
                            std::map<int, cv::Mat> *ptScoreMaps=NULL);
    /// Pursues parts greedily
    void pursueParts(const AppParam & rootw,
                     int numParts, int partWd, int partHt, int partScale,
                     vector<pair<AppParam, Anchor> > & parts);

  private:
    GrammarParam gParam_;
    FeatureParam featureParam_;
    Cell bgmu_;

    vector<Node *>  nodeSet_;  // Maintaining the graph structure
    vector<Edge *>  edgeSet_;
    Node * rootNode_;

    vector<Node *>  nodeDFS_; // Depth-First-Search ordering of nodes
    vector<Node *>  nodeBFS_; // Breadth-First-Search ordering of nodes
    // Depth-First-Search ordering of nodes for each component
    vector<vector<Node *> >  compNodeDFS_;
    // Breadth-First-Search ordering of nodes
    vector<vector<Node *> >  compNodeBFS_;

    vector<Appearance *>  appearanceSet_; // All parameters
    vector<Bias *> biasSet_;
    vector<Deformation *> deformationSet_;
    vector<Scaleprior *> scalepriorSet_;
    Rectangle maxDetectWindow_;
    Rectangle minDetectWindow_;

    Scalar thresh_;

    //(nodeId, predModel) for each object component,
    // 4-column matrix (x1, y1, x2, y2)
    std::map<int, Matrix> bboxPred_;

    /// for fast computing the alpha-processes
    vector<Filter *>  cachedFilters_;
    bool cached_; 

    /// AOGrid for seeking part configurations
    vector<AOGrid *>  gridAOG_;

    /// utility members used in save and read
    int idxRootNode_;

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;

}; /// class AOGrammar


/// Serialize a vector of models
template<int Dimension>
void save(const string & modelFile,
          const vector<AOGrammar > & models,
          int archiveType = 0);

template<int Dimension>
bool load(const string & modelFile,
          vector<AOGrammar > & models,
          int archiveType = 0);

} // namespace RGM

/// Somehow, I can not serialize a vector of AOGrammar without the following codes
namespace boost {
namespace serialization {
template <class Archive, int Dimension>
void save(Archive & ar, const std::vector<RGM::AOGrammar > & models,
          const unsigned int version) {
    ar.register_type(static_cast<RGM::AOGrammar *>(NULL));
    ar.template register_type<RGM::AOGrammar >();

    int num = models.size();

    ar & BOOST_SERIALIZATION_NVP(num);

    for(int i = 0;  i < num; ++i) {
        ar & models[i];
    }

}

template <class Archive, int Dimension>
void load(Archive & ar, std::vector<RGM::AOGrammar > & models,
          const unsigned int version) {
    ar.register_type(static_cast<RGM::AOGrammar *>(NULL));
    ar.template register_type<RGM::AOGrammar >();

    int num;

    ar & BOOST_SERIALIZATION_NVP(num);

    models.resize(num);
    for(int i = 0;  i < num; ++i) {
        ar & models[i];
    }
}

template <class Archive, int Dimension>
void serialize(Archive & ar, std::vector<RGM::AOGrammar > & models,
               const unsigned int version) {
    split_free(ar, models, version);
}

} // namespace serialization
} // namespace boost

#endif // RGM_AOGRAMMAR_HPP_
