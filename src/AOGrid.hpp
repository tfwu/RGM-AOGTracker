#ifndef RGM_AOGRID_HPP_
#define RGM_AOGRID_HPP_

#include "rgm_struct.hpp"

namespace RGM {

/// The AOGrid defines a Directed Acyclic And-Or Graph
/// which is used to quantize a grid (e.g., a HOG feature grid with W * H cells
/// or an image patch) to explore/unfold the space of latent structures
/// (part configurations) of objects or scenes
class AOGrid {

  public:
    /// A primitive type is represented by a rectangular shape used to tile
    /// the grid and defined by its width and height in a given grid
    typedef Rectangle  GridPrimitive;
    /// The dictionary of grid primitive types consists of all possible grid
    /// primtives enumerated in a given grid with some constraints satisfied
    /// (i.e., minimum side length/area of a grid primtive type)
    typedef vector<GridPrimitive> GridPrimitiveDict;

    /// A grid primitive instance is an instance by placing a grid primitive
    /// type at some valid location (x, y) in the grid,
    /// so it is defined by (x, y, wd, ht)
    class GridPrimitiveInstance : public GridPrimitive {
      public:
        /// Default constructor
        GridPrimitiveInstance() { init(); }
        /// Constructs an instance with given inputs
        GridPrimitiveInstance(const GridPrimitive & bbox, int idx) ;

        void init();
        void setBbox(const GridPrimitive & bbox) ;

        int dictIdx_;  // index in PrimitiveDict

        DEFINE_SERIALIZATION;
    };
    /// The set of primitive instances includes all instances of all primtive
    /// types placed in the grid
    typedef vector<GridPrimitiveInstance> GridPrimitiveInstanceSet;

    /// The node
    struct Vertex {
        /// The index of different node attributes
        enum {
            ID = 0, ID_IN_SUBSET, ID_IN_INSTANCE_SET,
            SPLIT_STEP1, SPLIT_STEP2,
            ID_BEST_CHILD, ID_G, ID_ON_OFF,
            IDX_NUM
        };

        typedef Eigen::Matrix<int, 1, IDX_NUM> Index;

        /// Default constructor
        Vertex() { init(); }

        /// Init
        void init();

        /// Computes the goodness
        void computeGoodness(int method, const vector<Scalar> & totalScores);

        /// Returns the scores
        const vector<Scalar> & scores(AOGrid & g, bool isPos) const;
        vector<Scalar> & getScores(AOGrid & g, bool isPos);

        /// Node description
        nodeType  type_;
        tangramSplitRule split_;
        Index     idx_;

        vector<int> childIDs_;
        vector<int> parentIDs_;

        long int numConfigurations_;

        vector<Scalar> pscores_;
        vector<pair<int, int> > pdxdy_;

        vector<Scalar> nscores_;
        vector<pair<int, int> > ndxdy_;

        Scalar goodness_;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        DEFINE_SERIALIZATION;
    };

    /// Default Constructor
    AOGrid() : rootTermNodeId_(-1), numAndNodes_(0), numOrNodes_(0),
        numTermNodes_(0), numConfigurations_(0) {}

    ~AOGrid() { clear(); }

    /// Constructs an AOGrid with given input args.
    AOGrid(const AOGridParam & param) : rootTermNodeId_(-1), numAndNodes_(0),
        numOrNodes_(0), numTermNodes_(0), numConfigurations_(0) {
        create(param);
    }

    /// Creates the AOGrid
    void create(const AOGridParam & param);

    /// Clears the AOGrid
    void clear();

    /// Returns if it is empty
    bool empty() const { return nodeSet_.empty(); }

    /// Parse
    void parse();

    vector<int> parseTreeBFS();

    bool findEqualBestChildForRootOr(Scalar goodnessThr);

    /// Visualize
    /// @param[in] nodeLabel: 1 - numConfig, 2 - DFS, 3 - BFS
    string visualize(string & saveDir, cv::Size sz, int edgeDirection = 0,
                     int nodeLabel = 0);

    /// Saves to a stream
    /// @param[in] archiveType 0 binary, 1 text
    void save(const string & modelFile, int archiveType = 0);

    /// Reads from a stream
    bool read(const string & modelFile, int archiveType = 0);

    /// Returns the param
    const AOGridParam  & param() const { return param_; }
    AOGridParam  & getParam() { return param_; }

    /// Returns the node set
    const vector<Vertex> & nodeSet() const { return nodeSet_; }
    vector<Vertex> & getNodeSet() { return nodeSet_; }

    /// Returns the root term node id
    int   rootTermNodeId() const { return rootTermNodeId_; }
    int & getRootTermNodeId() { return rootTermNodeId_; }

    /// Returns the number of nodes
    int   numAndNodes() const { return numAndNodes_; }
    int & getNumAndNodes() { return numAndNodes_; }

    int   numOrNodes() const { return numOrNodes_; }
    int & getNumOrNodes() { return numOrNodes_; }

    int   numTermNodes() const { return numTermNodes_; }
    int & getNumTermNodes() { return numTermNodes_; }

    /// Returns the DFS / BFS
    const vector<int> & DFSqueue() const { return DFSqueue_; }
    vector<int> & getDFSqueue() { return DFSqueue_; }

    const vector<int> & BFSqueue() const { return BFSqueue_; }
    vector<int> & getBFSqueue() { return BFSqueue_; }

    /// Returns the primitive dictionary
    const GridPrimitiveDict & dict() const { return dict_; }
    GridPrimitiveDict & getDict() { return dict_; }

    /// Returns the primitive instance set
    const GridPrimitiveInstanceSet & instanceSet() const { return instanceSet_; }
    GridPrimitiveInstanceSet & getInstanceSet() { return instanceSet_; }

    /// Returns the bounding box of an primitive instance
    /// in the input coordinate (not the grid coord.)
    Rectangle instanceBbox(int idx) const;

    /// Returns the number of configurations
    long int   numConfigurations() const { return numConfigurations_; }
    long int & getNumConfigurations() { return numConfigurations_; }

    /// Returns the scores
    const vector<Scalar> & pscores() const {  return pscores_; }
    vector<Scalar> & getPscores() {  return pscores_; }

    const vector<pair<int, int> > & pdxdy() const { return pdxdy_; }
    vector<pair<int, int> > & getPdxdy() { return pdxdy_; }

    /// Turns on/off all nodes
    void turnOnOff(bool isOn);

  private:
    void build();

    bool addNode(Vertex & node);
    bool isSameNode(const Vertex & node1, const Vertex & node2);
    int  findInstance(GridPrimitiveInstance & instance);
    int  updateDict(GridPrimitiveInstance & instance);
    void assignParentIDs();

    void traceDFS(int nodeID, vector<int> & visited);
    void traceBFS(int nodeID, vector<int> & visited);

    void countConfigurations();
    long int getDoubleCountedConfig(int nodeID);
    long int countDoubleCountings(
        vector<vector<int>::const_iterator > & combination);

    void pictureNodes(string & saveDir);
    void writeGraphVizDotFile(string & saveDir, cv::Size sz, int edgeDirection,
                              int nodeLabel);

  private:
    AOGridParam   param_;

    vector<Vertex> nodeSet_;
    int    rootTermNodeId_; // -1 if allowGridTermNode_ == false
    int    numAndNodes_;
    int    numOrNodes_;
    int    numTermNodes_;

    /// The depth-first and breadth-first visiting order of the nodes
    vector<int> DFSqueue_;
    vector<int> BFSqueue_;

    GridPrimitiveDict         dict_; // in the grid coordinate
    GridPrimitiveInstanceSet  instanceSet_;

    /// The number of configurations generated from the AOGrid
    long int  numConfigurations_;

    /// Whole template scores on positives
    vector<Scalar> pscores_;
    vector<pair<int, int> > pdxdy_;

    DEFINE_SERIALIZATION;

    DEFINE_RGM_LOGGER;
};


} // namespace RGM

#endif // RGM_AOGRID_HPP_
