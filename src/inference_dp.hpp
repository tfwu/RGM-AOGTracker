#ifndef RGM_INFERENCE_DP_HPP_
#define RGM_INFERENCE_DP_HPP_

#include "AOGrammar.hpp"

namespace RGM {
/// Predeclaration
template<int Dimension>
class ParseTree_;

/// Parsing with AOGrammar_ using DP
template<int Dimension>
class DPInference_ {

  public:
    /// Typedef
    typedef std::map<boost::uuids::uuid, vector<Matrix > >    Maps;
    typedef std::map<boost::uuids::uuid, vector<bool > >      Status;
    typedef std::map<boost::uuids::uuid, vector<MatrixXi> >   ArgMaps;
    typedef Detection_<Scalar> Detection;
    typedef vector<vector<vector<Matrix> > > OverlapMaps;
    typedef typename Appearance::Param AppParam;

    // constructor
    DPInference_() : grammar_(NULL), param_(NULL) {}

    explicit DPInference_(AOGrammar_<Dimension> & g, InferenceParam & p) :
        grammar_(&g), param_(&p) {}

    /// Sets grammar and inference param
    void setGrammar(AOGrammar_<Dimension> & g, InferenceParam & p) {
        grammar_ = &g;
        param_ = &p;
    }

    /// Computes the detection results
    void runDetection(const Scalar thresh, cv::Mat img, Scalar & maxDetNum,
                      vector<ParseTree> & pt);

    void runDetection(const Scalar thresh, const FeaturePyr & pyramid,
                      Scalar & maxDetNum, vector<ParseTree> & pt);

    /// Computes the score maps using DP algorithm
    bool runDP(const FeaturePyr & pyramid);

    /// Runs parsing
    void runParsing(const Scalar thresh, const FeaturePyr & pyramid,
                    Scalar & maxDetNum, vector<ParseTree > & pt);


    /// used in learning
    int runParsing(const Scalar thresh, const FeaturePyr & pyramid,
                   vector<ParseTree *> & pt, int startIdx, int endIdx);

    bool runParsing(const Scalar thresh, const FeaturePyr & pyramid,
                    ParseTree & pt);

    /// Computes a parse tree
    void parse(const FeaturePyr & pyramid, Detection & cand,
               ParseTree & pt, bool getLoss = false);

    int parse(const FeaturePyr & pyramid, Detection & cand,
              ParseTree & pt, Node * start, bool getLoss = false);

    /// Computes overlap maps
    /// @param[in] overlapMaps It will return overlap maps for
    ///  each level in the score maps, each box in @p bboxes and
    ///  each object AND-node
    /// @param[out] the valid state of levels
    vector<bool> computeOverlapMaps(const vector<Rectangle> & bboxes,
                                    const FeaturePyr & pyr,
                                    OverlapMaps & overlapMaps,
                                    Scalar overlapThr);

    /// Copies score maps of a node
    void copyScoreMaps(const Node * n);

    /// Recovers score maps of a node
    void recoverScoreMaps(const Node * n);

    /// Applies output inhibition of a given bounding box
    /// Inhibit all detection window locations that do not yield sufficient
    /// overlap with the given bounding box by setting the score = -inf
    void inhibitOutput(int idxBox, OverlapMaps & overlapMap,
                       Scalar overlapThr, bool needCopy);

    void inhibitAllFg(OverlapMaps & overlapMap, Scalar overlapThr, bool needCopy);

    void inhibitAllFg(OverlapMaps & overlapMap, Scalar minOvThr, Scalar maxOvThr,
                      bool needCopy);

    /// Applies loss adjustment
    void applyLossAdjustment(int idxBox, int nbBoxes, OverlapMaps & overlapMap,
                             Scalar fgOverlap, Scalar bgOverlap, bool needCopy);

    /// Gets the score maps of a parse tree
    void getPtScoreMaps(ParseTree & pt, std::map<int, Mat> & maps,
                        int cropSz, string saveName = "",
                        std::map<int, Scalar> * fom = NULL);

    /// Visualizes all the score maps
    void visScoreMaps(string & saveDir, ParseTree * pt = NULL);

    /// Computes the quality of Tnodes using intrackability
    vector<int> computeIntrackability(ParseTree & pt,
                                      std::map<int, Scalar> & out,
                                      Scalar thr = 1.0F);
    Scalar computeIntrackability(ParseTree &pt);

    /// Computes filter responses of T-nodes, i.e., alpha-processes
    bool computeAlphaProcesses(const FeaturePyr & pyramid);

    /// Computes the scale prior feature
    void computeScalePriorFeature(int nbLevels);

    /// Applies the compositional rule or the deformation rule for an AND-node
    bool computeANDNode(Node * node, int padx, int pady);

    /// Bounded DT
    void DT2D(Matrix & scoreMap, Deformation::Param & w, int shift,
              MatrixXi & Ix, MatrixXi & Iy);
    void DT1D(const Scalar * vals, Scalar * out_vals, int * I, int step,
              int n, Scalar a, Scalar b, int shift,
              int * v, Scalar * z, Scalar * t);

    /// Normal DT
    void DT2D(Matrix & scoreMap, Deformation::Param & w, MatrixXi & Ix,
              MatrixXi & Iy);
    void DT1D(const Scalar * x, int n, Scalar a, Scalar b, Scalar * z, int * v,
              Scalar * y, int * m, const Scalar * t, int incx, int incy,
              int incm);

    /// Applies the switching rule for an OR-node
    bool computeORNode(Node * node);

    /// Parses an OR-node
    bool parseORNode(int head, vector<Node * > & gBFS, vector<int> & ptBFS,
                     const FeaturePyr & pyramid, ParseTree & pt, bool getLoss);

    /// Parses an AND-node
    bool parseANDNode(int head, vector<Node *> & gBFS, vector<int> & ptBFS,
                      const FeaturePyr & pyramid, ParseTree & pt);

    /// Parses an T-node
    bool parseTNode(int idx, vector<Node *> & gBFS, vector<int> & ptBFS,
                    const FeaturePyr & pyramid, ParseTree & pt);

    /// Returns score maps of a node
    const vector<Matrix > & scoreMaps(const Node * n);
    vector<Matrix > & getScoreMaps(const Node * n);

    /// Returns copied score maps of a node
    const vector<Matrix > & scoreMapCopies(const Node * n);
    vector<Matrix > & getScoreMapCopies(const Node * n);

    /// Returns status of score maps of a node
    const vector<bool> & scoreMapStatus(const Node * n);
    vector<bool> & getScoreMapStatus(const Node * n);

    /// Sets the status of score maps of a node
    void setScoreMapStatus(const Node * n, int l);

    /// Set score maps to a node
    void setScoreMaps(const Node * n, int nbLevels, vector<Matrix> & s,
                      const vector<bool> & validLevels);

    /// Returns the deformation maps
    vector<MatrixXi> & getDeformationX(const Node * n);
    vector<MatrixXi> & getDeformationY(const Node * n);

    /// Returns loss maps of a node
    const vector<Matrix > & lossMaps(const Node * n);
    vector<Matrix > & getLossMaps(const Node * n);

    /// Release memory
    void release();

  private:
    AOGrammar * grammar_;
    InferenceParam * param_;

    Matrix scalepriorFeatures_;     // a 3 * nbLevels matrix

    // for nodes in an AOG, for each level in the feature pyramid
    Maps scoreMaps_;
    Maps scoreMapCopies_;
    Status scoreMapStatus_;
    ArgMaps deformationX_;
    ArgMaps deformationY_;
    Maps lossMaps_;

    DEFINE_RGM_LOGGER;
};
} // namespace RGM
#endif // RGM_INFERENCE_DP_HPP_
