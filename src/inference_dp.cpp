#include "inference_dp.hpp"
#include "parse_tree.hpp"
#include "util/UtilFile.hpp"
#include "util/UtilGeneric.hpp"
#include "util/UtilOpencv.hpp"
#include "util/UtilString.hpp"

namespace RGM {

// ------- DPInference_ -------

template<int Dimension>
void DPInference::runDetection(const Scalar thresh, Mat img, Scalar & maxDetNum,
                               vector<ParseTree> & pt) {
    if(maxDetNum <= 0)
        return;

    AOGrammar & g(*grammar_);

    FeaturePyr pyr(img, g.featParam(), g.bgmu());
    if(pyr.empty())
        return;

    runDetection(thresh, pyr, maxDetNum, pt);
}

template<int Dimension>
void DPInference::runDetection(const Scalar thresh, const FeaturePyr & pyramid,
                               Scalar & maxDetNum, vector<ParseTree> & pt) {
    if(maxDetNum <= 0)
        return;

    if(!runDP(pyramid))
        return;

    runParsing(thresh, pyramid, maxDetNum, pt);
}

template<int Dimension>
bool DPInference::runDP(const FeaturePyr & pyramid) {
    // compute score maps for T-nodes
    if(!computeAlphaProcesses(pyramid)) {
        RGM_LOG(error, "Failed to computer filter responses.");
        return false;
    }

    computeScalePriorFeature(pyramid.nbLevels());

    // Using DFS order
    vector<Node *> & nDFS(grammar_->getNodeDFS());

    for(int i = 0; i < nDFS.size(); ++i) {
        Node * curNode = nDFS[i];
        nodeType t = curNode->type();

        switch(t) {
            case AND_NODE: {
                if(!computeANDNode(curNode, pyramid.padx(), pyramid.pady())) {
                    RGM_LOG(error, "Failed to compute an And-node.");
                    return false;
                }
                break;
            }
            case OR_NODE: {
                if(!computeORNode(curNode)) {
                    RGM_LOG(error, "Failed to compute an Or-node.");
                    return false;
                }

                break;
            }
        }         // switch t
    }         // for i

    return true;
}

template<int Dimension>
void DPInference::runParsing(const Scalar thresh, const FeaturePyr & pyramid,
                             Scalar & maxDetNum, vector<ParseTree> & pt) {
    // Find scores above the threshold
    vector<Detection> cands;
    for(int level = 0; level < pyramid.nbLevels(); ++level) {
        if(!pyramid.validLevels()[level])
            continue;

        const Matrix & score(scoreMaps(grammar_->rootNode())[level]);
        const int rows = score.rows();
        const int cols = score.cols();

        for(int y = 0; y < score.rows(); ++y)
            for(int x = 0; x < score.cols(); ++x) {
                const Scalar s = score(y, x);
                if(s > thresh) {
                    // Non-maxima suppresion in a 3x3 neighborhood
//                    if(((y == 0) || (x == 0) || (s > score(y - 1, x - 1))) &&
//                            ((y == 0) || (s > score(y - 1, x))) &&
//                            ((y == 0) || (x == cols - 1) || (s > score(y - 1, x + 1))) &&
//                            ((x == 0) || (s > score(y, x - 1))) &&
//                            ((x == cols - 1) || (s > score(y, x + 1))) &&
//                            ((y == rows - 1) || (x == 0) || (s > score(y + 1, x - 1))) &&
//                            ((y == rows - 1) || (s > score(y + 1, x))) &&
//                            ((y == rows - 1) || (x == cols - 1) || (s > score(y + 1, x + 1))))
                    {
                        // here, (x, y) is in the coordinate with padding
                        cands.push_back(Detection(level, x, y, s));
                    }
                }
            }
    }

    if(cands.empty())
        //LOG(INFO) << "Not found detections";
        return;

    // Sort scores in descending order
    std::sort(cands.begin(), cands.end());

    if(cands.size() > maxDetNum)
        cands.resize(maxDetNum);

    bool getLoss = lossMaps(grammar_->nodeBFS()[1]).size();

    // Compute detection windows, filter bounding boxes, and derivation trees
    int numDet = cands.size();
    pt.resize(numDet);

    #pragma omp parallel for
    for(int i = 0; i < numDet; ++i) {
        parse(pyramid, cands[i], pt[i], getLoss);
        if(param_->useNMS_) {
            ParseInfo * info = pt[i].getParseInfo(pt[i].rootNode());
            if(info->clipBbox(pyramid.imgWd(), pyramid.imgHt(),
                              param_->clipPt_)) {
                // use c_ to record the index which will be used to
                // select the pt after NMS
                cands[i].c_ = i;
                cands[i].setX(info->x());
                cands[i].setY(info->y());
                cands[i].setWidth(info->width());
                cands[i].setHeight(info->height());
            } else  {
                cands[i].c_ = -1;
                cands[i].score_ = -Inf;
            }
        }
    }

    if(param_->useNMS_) {
        std::sort(cands.begin(), cands.end());

        for(int i = 1; i < cands.size(); ++i)
            cands.resize(std::remove_if(
                             cands.begin() + i, cands.end(),
                             Intersector_<Scalar>(cands[i - 1],
                                                  param_->nmsOverlap_,
                                                  param_->nmsDividedByUnion_)) -
                         cands.begin());

        vector<ParseTree> ptNMS;
        ptNMS.reserve(cands.size());

        for(int i = 0; i < cands.size(); ++i) {
            if(cands[i].c_ == -1)
                break;

            int idx = cands[i].c_;
            ptNMS.push_back(pt[idx]);
        }

        pt.swap(ptNMS);
    }
}

template<int Dimension>
int DPInference::runParsing(const Scalar thresh,
                            const FeaturePyr & pyramid,
                            vector<ParseTree *> & pt,
                            int startIdx, int endIdx) {
    // Find scores above the threshold
    vector<Detection> cands;
    for(int level = 0; level < pyramid.nbLevels(); ++level) {
        if(!pyramid.validLevels()[level])
            continue;
        const Matrix & score(scoreMaps(grammar_->rootNode())[level]);
        for(int y = 0; y < score.rows(); ++y)
            for(int x = 0; x < score.cols(); ++x) {
                const Scalar s = score(y, x);
                if(s > thresh)
                    // Non-maxima suppresion in a 3x3 neighborhood
                    // here, (x, y) is in the coordinate with padding
                    cands.push_back(Detection(level, x, y, s));
            }
    }

    if(cands.empty())
        return 0;

    // Sort scores in descending order
    std::sort(cands.begin(), cands.end());

    int numToGet = endIdx - startIdx;
    if(cands.size() > numToGet)
        cands.resize(numToGet);

    bool getLoss = lossMaps(grammar_->nodeBFS()[1]).size();

    // Compute detection windows, filter bounding boxes, and derivation trees
    int numDet = cands.size();
    #pragma omp parallel for
    for(int i = 0; i < numDet; ++i)
        parse(pyramid, cands[i], *pt[startIdx + i], getLoss);

    return numDet;
}

template<int Dimension>
bool DPInference::runParsing(const Scalar thresh, const FeaturePyr & pyramid,
                             ParseTree & pt) {
    // Find scores above the threshold
    vector<Detection> cands;
    for(int level = 0; level < pyramid.nbLevels(); ++level) {
        if(!pyramid.validLevels()[level])
            continue;
        const Matrix & score(scoreMaps(grammar_->rootNode())[level]);
        for(int y = 0; y < score.rows(); ++y)
            for(int x = 0; x < score.cols(); ++x) {
                const Scalar s = score(y, x);
                if(s > thresh)
                    // here, (x, y) is in the coordinate with padding
                    cands.push_back(Detection(level, x, y, s));
            }
    }

    if(cands.empty())
        return false;

    // Sort scores in descending order
    std::sort(cands.begin(), cands.end());

    bool getLoss = lossMaps(grammar_->nodeBFS()[1]).size();

    // Compute detection windows, filter bounding boxes, and derivation trees
    parse(pyramid, cands[0], pt, getLoss);

    return true;
}

template<int Dimension>
void DPInference::parse(const FeaturePyr & pyramid, Detection & cand,
                        ParseTree & pt, bool getLoss) {
    pt.clear();
    pt.setGrammar(*grammar_);

    pt.getImgWd() = pyramid.imgWd();
    pt.getImgHt() = pyramid.imgHt();

    // Backtrack solution in BFS
    vector<Node *> gBFS;
    gBFS.push_back(grammar_->getRootNode());

    // Get the parse info for the root node
    // note that cand.(x_, y_) are in the coordinate with padding
    ParseInfo pinfo(-1, cand.l_, cand.x_, cand.y_, 0, 0, 0, cand.score_,
                    Rectangle_<Scalar>());
    int idxInfo = pt.addParseInfo(pinfo);

    // Add the root node to pt
    int t = static_cast<int>(grammar_->rootNode()->type());
    int gNode = grammar_->idxNode(grammar_->rootNode());
    pt.getIdxRootNode() = pt.addNode(gNode, t);
    pt.getRootNode()->getIdx()[PtNode::IDX_PARSEINFO] = idxInfo;

    // BFS for pt
    vector<int> ptBFS;
    ptBFS.push_back(pt.idxRootNode());

    int head = 0;
    while(head < gBFS.size()) {
        Node * curNode = gBFS[head];
        nodeType t = curNode->type();

        switch(t) {
            case T_NODE: {
                if(!parseTNode(head, gBFS, ptBFS, pyramid, pt))
                    return;

                break;
            }
            case AND_NODE: {
                if(!parseANDNode(head, gBFS, ptBFS, pyramid, pt))
                    return;

                break;
            }
            case OR_NODE: {
                if(!parseORNode(head, gBFS, ptBFS, pyramid, pt, getLoss))
                    return;

                break;
            }
            default: {
                RGM_LOG(error, "Wrong type of nodes.");
                return;
            }
        }         // switch

        head++;
    }         // while
}

template<int Dimension>
int DPInference::parse(const FeaturePyr & pyramid, Detection & cand,
                       ParseTree & pt, Node * start, bool getLoss) {

    RGM_CHECK_NOTNULL(start);

    if(start == grammar_->getRootNode()) {
        pt.clear();
        pt.setGrammar(*grammar_);

        pt.getImgWd() = pyramid.imgWd();
        pt.getImgHt() = pyramid.imgHt();
    }

    // Backtrack solution in BFS
    vector<Node *> gBFS;
    gBFS.push_back(start);

    // Get the parse info for the root node
    // note that cand.(x_, y_) are in the coordinate with padding
    ParseInfo pinfo(-1, cand.l_, cand.x_, cand.y_, 0, 0, 0, cand.score_,
                    Rectangle_<Scalar>());
    int idxInfo = pt.addParseInfo(pinfo);

    // Add the root node to pt
    int t = static_cast<int>(start->type());
    int gNode = grammar_->idxNode(start);

    int idx = pt.addNode(gNode, t);
    if(start == grammar_->getRootNode())
        pt.getIdxRootNode() = idx;

    pt.getNodeSet()[idx]->getIdx()[PtNode::IDX_PARSEINFO] = idxInfo;

    // BFS for pt
    vector<int> ptBFS;
    ptBFS.push_back(pt.idxRootNode());

    int head = 0;
    while(head < gBFS.size()) {
        Node * curNode = gBFS[head];
        nodeType t = curNode->type();

        switch(t) {
            case T_NODE: {
                if(!parseTNode(head, gBFS, ptBFS, pyramid, pt))
                    return -1;

                break;
            }
            case AND_NODE: {
                if(!parseANDNode(head, gBFS, ptBFS, pyramid, pt))
                    return -1;

                break;
            }
            case OR_NODE: {
                if(!parseORNode(head, gBFS, ptBFS, pyramid, pt, getLoss))
                    return -1;

                break;
            }
            default: {
                RGM_LOG(error, "Wrong type of nodes.");
                return -1;
            }
        }         // switch

        head++;
    }         // while

    return idx;
}

template<int Dimension>
bool DPInference::computeAlphaProcesses(const FeaturePyr & pyramid) {
    // Transform the filters if needed
    #pragma omp critical
    if(!grammar_->cachedFFTStatus())
        grammar_->cachingFilters();

    while(!grammar_->cachedFFTStatus())
        RGM_LOG(normal, "Waiting for caching the FFT filters");

    // Create a patchwork
    const Patchwork patchwork(pyramid);

    // Convolve the patchwork with the filters
    int nbFilters = grammar_->cachedFilters().size();
    // per Appearance per valid Level
    vector<vector<Matrix> > filterResponses(nbFilters);

    patchwork.convolve(grammar_->cachedFilters(), filterResponses);

    if(filterResponses.empty()) {
        RGM_LOG(error, "filter convolution failed.");
        return false;
    }

    int nbLevel = pyramid.nbLevels();
    int nbValidLevel = filterResponses[0].size();
    RGM_CHECK_EQ(nbValidLevel, pyramid.nbValidLevels());

    // score maps of root node
    vector<Matrix> & rootScoreMaps(getScoreMaps(grammar_->rootNode()));
    rootScoreMaps.resize(nbLevel);

    // Normalize the sizes of filter response maps per level
    for(int l = 0, ll = 0; l < nbLevel; ++l) {
        if(pyramid.validLevels()[l]) {
            int maxHt = 0;
            int maxWd = 0;
            for(int i = 0; i < nbFilters; ++i) {
                maxHt = std::max<int>(maxHt, filterResponses[i][ll].rows());
                maxWd = std::max<int>(maxWd, filterResponses[i][ll].cols());
            }

            for(int i = 0; i < nbFilters; ++i) {
                Matrix tmp = Matrix::Constant(maxHt, maxWd, -Inf);
                tmp.block(0, 0,
                          filterResponses[i][ll].rows(),
                          filterResponses[i][ll].cols()) = filterResponses[i][ll];

                filterResponses[i][ll].swap(tmp);
            }

            rootScoreMaps[l] = Matrix::Zero(maxHt, maxWd);

            ++ll;
        } else
            rootScoreMaps[l] = Matrix::Zero(1, 1);
    }

    // Assign to T-nodes
    for(int i = 0, t = 0; i < grammar_->nodeSet().size(); ++i)
        if(grammar_->nodeSet()[i]->type() == T_NODE &&
                grammar_->nodeSet()[i]->onOff()) {
            setScoreMaps(grammar_->nodeSet()[i], nbLevel, filterResponses[t],
                         pyramid.validLevels());
            ++t;
        }

    return true;
}

template<int Dimension>
void DPInference::computeScalePriorFeature(int nbLevels) {
    Scaleprior::Param tmp;
    scalepriorFeatures_ = Matrix::Zero(tmp.cols(), nbLevels);

    int s = 0;
    int e = std::min<int>(nbLevels, grammar_->interval());
    scalepriorFeatures_.block(0, s, 1, e).fill(1);

    s = e;
    e = std::min<int>(nbLevels, e * 2);
    scalepriorFeatures_.block(1, s, 1, e - s).fill(1);

    s = e;
    scalepriorFeatures_.block(2, s, 1, nbLevels - s).fill(1);
}

template<int Dimension>
bool DPInference::computeANDNode(Node * node, int padx, int pady) {
    if(node == NULL || node->type() != AND_NODE) {
        RGM_LOG(error, "Need a valid AND-node as input.");
        return false;
    }

    if(node->outEdges().size() == 1 &&
            node->outEdges()[0]->type() == TERMINATION)
        return true;

    if(node->outEdges().size() == 1 &&
            node->outEdges()[0]->type() == DEFORMATION) {
        // deformation rule -> apply distance transform
        Deformation::Param w = node->deformationParam();

        // init the score maps using those of the toNode
        vector<Matrix> & score(getScoreMaps(node));
        score = scoreMaps(node->outEdges()[0]->toNode());

        int nbLevel = score.size();

        vector<MatrixXi> & x(getDeformationX(node));
        vector<MatrixXi> & y(getDeformationY(node));

        x.resize(nbLevel);
        y.resize(nbLevel);

        #pragma omp parallel for
        for(int i = 0; i < nbLevel; ++i)
            // Bounded distance transform with +/- 4 HOG cells (9x9 window)
            DT2D(score[i], w, Deformation::BoundedShiftInDT, x[i], y[i]);

        return true;
    }

    RGM_CHECK_EQ(node->outEdges()[0]->type(), COMPOSITION);

    // composition rule -> shift and sum scores from toNodes
    vector<Matrix> & score(getScoreMaps(node));
    score = scoreMaps(grammar_->rootNode());

    int nbLevels = score.size();

    // prepare score for this rule
    Scalar bias = 0;
    if(node->bias() != NULL)
        bias = node->bias()->w() * grammar_->featureBias();

    Scaleprior::Vector scalePriorScore = Scaleprior::Vector::Zero(nbLevels);
    if(node->scaleprior() != NULL)
        scalePriorScore = node->scaleprior()->w() * scalepriorFeatures_;

    for(int i = 0; i < nbLevels; ++i)
        score[i].fill(bias + scalePriorScore(i));

    // sum scores from toNodes (with appropriate shift and down sample)
    vector<Edge *> & outEdges = node->getOutEdges();
    for(int i = 0; i < outEdges.size(); ++i) {
        if(!outEdges[i]->onOff() ||
                !outEdges[i]->toNode()->onOff())
            continue;
        const Anchor & curAnchor = outEdges[i]->toNode()->anchor();
        int ax = curAnchor(0);
        int ay = curAnchor(1);
        int ds = curAnchor(2);

        // step size for down sampling
        int step = std::pow(2.0f, ds);

        // amount of (virtual) padding to hallucinate
        int virtpady = (step - 1) * pady;
        int virtpadx = (step - 1) * padx;

        // starting points (simulates additional padding at finer scales)
        // @note (ax, ay) are computed without considering padding.
        // So, given a root location (x, y) in the score map (computed with padded feature map)
        // the location of a part will be: (x-padx) * step + ax without considering padding
        // and (x-padx) * step + ax + padx = x + [ax - (step-1)*padx]
        int starty = ay - virtpady;
        int startx = ax - virtpadx;

        // score table to shift and down sample
        const vector<Matrix> & s(scoreMaps(outEdges[i]->toNode()));

        for(int j = 0; j < s.size(); ++j) {
            int level = j - grammar_->interval() * ds;
            if(level >= 0) {
                // ending points
                int endy = min<int>(s[level].rows(),
                                    starty + step * (score[j].rows() - 1) + 1);
                int endx = min<int>(s[level].cols(),
                                    startx + step * (score[j].cols() - 1) + 1);

                // y sample points
                vector<int> iy;
                int oy = 0;
                for(int yy = starty; yy < endy; yy += step) {
                    if(yy < 0)
                        oy++;
                    else
                        iy.push_back(yy);
                }

                // x sample points
                vector<int> ix;
                int ox = 0;
                for(int xx = startx; xx < endx; xx += step) {
                    if(xx < 0)
                        ox++;
                    else
                        ix.push_back(xx);
                }

                // sample scores
                Matrix sp(iy.size(), ix.size());
                for(int yy = 0; yy < iy.size(); ++yy)
                    for(int xx = 0; xx < ix.size(); ++xx)
                        sp(yy, xx) = s[level](iy[yy], ix[xx]);

                // sum with correct offset
                Matrix stmp = Matrix::Constant(score[j].rows(), score[j].cols(),
                                               -Inf);
                stmp.block(oy, ox, sp.rows(), sp.cols()) = sp;
                score[j] += stmp;
            } else
                score[j].fill(-Inf);
        }
    }

    return true;
}

template<int Dimension>
void DPInference::DT2D(Matrix & scoreMap, Deformation::Param & w,
                       int shift, MatrixXi & Ix, MatrixXi & Iy) {
    Scalar ax = w(0);            // dx^2 dx dy^2 dy
    Scalar bx = w(1);
    Scalar ay = w(2);
    Scalar by = w(3);

    const int rows = static_cast<int>(scoreMap.rows());
    const int cols = static_cast<int>(scoreMap.cols());

    Matrix tmpOut = Matrix::Zero(rows, cols);
    MatrixXi tmpIy = MatrixXi::Zero(rows, cols);
    Ix = MatrixXi::Zero(rows, cols);
    Iy = MatrixXi::Zero(rows, cols);

    // Temporary vectors
    vector<Scalar> z(std::max<int>(rows, cols) + 1);
    vector<int> v(std::max<int>(rows, cols) + 1);
    vector<Scalar> t(std::max<int>(rows, cols));

    // cache divisive factors used in 1d distance transforms
    t[0] = Inf;
    for(int y = 1; y < rows; ++y)
        t[y] = 1 / (-ay * y);

    for(int x = 0; x < scoreMap.cols(); ++x)
        DT1D(scoreMap.col(x).data(), tmpOut.col(x).data(), tmpIy.col(x).data(),
             scoreMap.cols(), scoreMap.rows(), -ay, -by, shift, &v[0], &z[0],
             &t[0]);

    for(int x = 1; x < cols; ++x)
        t[x] = 1 / (-ax * x);

    for(int y = 0; y < scoreMap.rows(); ++y)
        DT1D(tmpOut.row(y).data(), scoreMap.row(y).data(), Ix.row(y).data(), 1,
             scoreMap.cols(), -ax, -bx, shift, &v[0], &z[0], &t[0]);

    // get argmax
    for(int x = 0; x < scoreMap.cols(); x++)
        for(int y = 0; y < scoreMap.rows(); y++)
            Iy(y, x) = tmpIy(y, Ix(y, x));
}

template<int Dimension>
void DPInference::DT1D(const Scalar * vals, Scalar * out_vals, int * I,
                       int step, int n, Scalar a, Scalar b,
                       int shift, int * v, Scalar * z, Scalar * t) {
    int k = 0;
    v[0] = 0;
    z[0] = -Inf;
    z[1] = Inf;

    Scalar aInv = 1 / a;

    for(int q = 1; q <= n - 1; q++) {
        // compute unbounded point of intersection
        Scalar s = 0.5 * ((vals[q * step] - vals[v[k] * step]) * t[q - v[k]]
                          + q + v[k] - b * aInv);

        // bound point of intersection; +/- eps to handle boundary conditions
        s = min<Scalar>(v[k] + shift + EPS, max<Scalar>(q - shift - EPS, s));

        while(s <= z[k]) {
            // delete dominiated parabola
            k--;
            s = 0.5 * ((vals[q * step] - vals[v[k] * step]) * t[q - v[k]]
                       + q + v[k] - b * aInv);
            s = min<Scalar>(v[k] + shift + EPS, max<Scalar>(q - shift - EPS, s));
        }
        k++;
        v[k] = q;
        z[k] = s;
    }
    z[k + 1] = Inf;

    k = 0;
    for(int q = 0; q < n; q++) {
        while(z[k + 1] < q)
            k++;
        out_vals[q * step] = a * std::pow(q - v[k], 2.0F) + b * (q - v[k]) +
                             vals[v[k] * step];
        I[q * step] = v[k];
    }

    //    for (int i = 0; i < n; i++) {
    //        Scalar max_val = -std::numeric_limits<Scalar>::infinity();
    //        int argmax     = 0;
    //        int first      = std::max<int>(0,   i-shift);
    //        int last       = std::min<int>(n-1, i+shift);
    //        for (int j = first; j <= last; j++) {
    //            Scalar val = vals[j*step] - a*(i-j)*(i-j) - b*(i-j);
    //            if (val > max_val) {
    //                max_val = val;
    //                argmax  = j;
    //            }
    //        }
    //        out_vals[i*step] = max_val;
    //        I[i*step] = argmax;
    //    }
}

template<int Dimension>
void DPInference::DT2D(Matrix & scoreMap, Deformation::Param & w,
                       MatrixXi & Ix, MatrixXi & Iy) {
    // Nothing to do if the matrix is empty
    if(!scoreMap.size())
        return;

    Scalar ax = w(0);            // dx^2 dx dy^2 dy
    Scalar bx = w(1);
    Scalar ay = w(2);
    Scalar by = w(3);

    const int rows = static_cast<int>(scoreMap.rows());
    const int cols = static_cast<int>(scoreMap.cols());

    Ix = MatrixXi::Zero(rows, cols);
    Iy = MatrixXi::Zero(rows, cols);

    Matrix tmp(rows, cols);

    // Temporary vectors
    vector<Scalar> z(std::max(rows, cols) + 1);
    vector<int> v(std::max(rows, cols) + 1);
    vector<Scalar> t(std::max(rows, cols));

    t[0] = std::numeric_limits<Scalar>::infinity();

    for(int x = 1; x < cols; ++x)
        t[x] = 1 / (-ax * x);

    // Filter the rows in tmp
    for(int y = 0; y < rows; ++y)
        DT1D(scoreMap.row(y).data(), cols, -ax, -bx, &z[0], &v[0],
             tmp.row(y).data(), Ix.row(y).data(), &t[0],
             1, 1, 1);

    for(int y = 1; y < rows; ++y)
        t[y] = 1 / (-ay * y);

    // Filter the columns back to the original matrix
    for(int x = 0; x < cols; ++x)
        DT1D(tmp.data() + x, rows, -ay, -by, &z[0], &v[0], 0, Iy.col(x).data(),
             &t[0], cols, cols, 1);

    // Re-index the best x positions now that the best y changed
    tmp = Ix.template cast<Scalar>();

    for(int y = 0; y < rows; ++y)
        for(int x = 0; x < cols; ++x)
            Ix(y, x) = static_cast<int>(tmp(Iy(y, x), x));
}

template<int Dimension>
void DPInference::DT1D(const Scalar * x, int n, Scalar a, Scalar b,
                       Scalar * z, int * v, Scalar * y,
                       int * m, const Scalar * t,
                       int incx, int incy, int incm) {
    RGM_CHECK(x && (y || m), error);
    RGM_CHECK_GT(n, 0);
    RGM_CHECK_LT(a, 0);
    RGM_CHECK(z && v, error);
    RGM_CHECK_NOTNULL(t);
    RGM_CHECK(incx && incy && (m ? incm : true), error);

    z[0] = -Inf;
    z[1] = Inf;
    v[0] = 0;

    // Use a lookup table to replace the division by (a * (i - v[k]))
    int k = 0;
    Scalar xvk = x[0];

    for(int i = 1; i < n;) {
        const Scalar s = (x[i * incx] - xvk) * t[i - v[k]] + (i + v[k]) - b / a;

        if(s <= z[k]) {
            --k;
            xvk = x[v[k] * incx];
        } else  {
            ++k;
            v[k] = i;
            z[k] = s;
            xvk = x[i * incx];
            ++i;
        }
    }

    z[k + 1] = Inf;

    if(y && m)
        for(int i = 0, k = 0; i < n; ++i) {
            while(z[k + 1] < 2 * i)
                ++k;

            y[i * incy] = x[v[k] * incx] + (a * (i - v[k]) + b) * (i - v[k]);
            m[i * incm] = v[k];
        }
    else if(y)
        for(int i = 0, k = 0; i < n; ++i) {
            while(z[k + 1] < 2 * i)
                ++k;

            y[i * incy] = x[v[k] * incx] + (a * (i - v[k]) + b) * (i - v[k]);
        }
    else
        for(int i = 0, k = 0; i < n; ++i) {
            while(z[k + 1] < 2 * i)
                ++k;

            m[i * incm] = v[k];
        }
}

template<int Dimension>
bool DPInference::computeORNode(Node * node) {
    if(node == NULL || node->type() != OR_NODE || !node->onOff()) {
        RGM_LOG(error, "Need valid OR-node as input.");
        return false;
    }

    if(node->outEdges().size() == 1)
        return true;

    // take pointwise max over scores of toNodes or outEdges
    vector<Matrix> & score(getScoreMaps(node));
    int i = 0;
    bool status = node->outEdges()[i]->onOff() &&
                  node->outEdges()[i]->toNode()->onOff();
    while(!status) {
        i++;
        status = node->outEdges()[i]->onOff() &&
                 node->outEdges()[i]->toNode()->onOff();
    }
    RGM_CHECK_LT(i, node->outEdges().size());

    score = scoreMaps(node->outEdges()[i]->toNode());
    for(++i; i < node->outEdges().size(); ++i) {
        if(!node->outEdges()[i]->onOff() ||
                !node->outEdges()[i]->toNode()->onOff())
            continue;
        for(int j = 0; j < score.size(); ++j)
            score[j] = score[j].cwiseMax(scoreMaps(
                                             node->outEdges()[i]->toNode())[j]);
    }         // for i

    return true;
}

template<int Dimension>
bool DPInference::parseORNode(int idx, vector<Node *> & gBFS,
                              vector<int> & ptBFS,
                              const FeaturePyr & pyramid, ParseTree & pt,
                              bool getLoss) {
    Node * gNode = gBFS[idx];
    if(gNode->type() != OR_NODE || !gNode->onOff()) {
        RGM_LOG(error, "Not an OR-node or Or-node is turned off.");
        return false;
    }

    int fromIdx = ptBFS[idx];
    PtNode * ptNode = pt.getNodeSet()[fromIdx];

    if(ptNode->getIdx()[PtNode::IDX_PARSEINFO] == -1) {
        RGM_LOG(error, "Need parse info. for the current pt node.");
        return false;
    }

    ParseInfo * info = pt.getParseInfo(ptNode);

    // Finds the best child of the OR-node by score matching
    int idxArgmax = -1;
    vector<Edge *> & outEdges(gNode->getOutEdges());
    for(int i = 0; i < outEdges.size(); ++i) {
        if(!outEdges[i]->onOff())
            continue;

        int y = info->y_ - VirtualPadding(pyramid.pady(), info->ds_);
        int x = info->x_ - VirtualPadding(pyramid.padx(), info->ds_);
        Scalar s = scoreMaps(outEdges[i]->toNode())[info->l_](y, x);
        if(info->score_ == s) {
            idxArgmax = i;
            break;
        }
    }         // for i

    if(idxArgmax == -1) {
        RGM_LOG(error, "Failed to find the best child.");
        return false;
    }

    // Get the best switching
    info->c_ = idxArgmax;
    Edge * bestEdge = outEdges[idxArgmax];
    Node * bestChild = bestEdge->getToNode();

    // Add an edge and a node to pt
    int idxG = grammar_->idxNode(bestChild);
    int t = static_cast<int>(bestChild->type());
    int toIdx = pt.addNode(idxG, t);
    PtNode * toNode = pt.getNodeSet()[toIdx];

    idxG = grammar_->idxEdge(bestEdge);
    int edge = pt.addEdge(fromIdx, toIdx, idxG, bestEdge->type());

    // Add the node to BFS
    gBFS.push_back(bestEdge->getToNode());
    ptBFS.push_back(toIdx);

    if(gNode == grammar_->getRootNode()) {
        const Rectangle & detectWind = bestChild->detectWindow();

        // Detection scale
        Scalar scale = static_cast<Scalar>(grammar_->cellSize()) /
                       pyramid.scales()[info->l_];

        // compute and record image coordinates of the detection window
        Scalar x1 = (info->x_ - pyramid.padx() * std::pow<int>(2, info->ds_)) *
                    scale;
        Scalar y1 = (info->y_ - pyramid.pady() * std::pow<int>(2, info->ds_)) *
                    scale;
        Scalar x2 = x1 + detectWind.width() * scale - 1;
        Scalar y2 = y1 + detectWind.height() * scale - 1;

        // update the parse info.
        info->setX(x1);
        info->setY(y1);
        info->setWidth(x2 - x1 + 1);
        info->setHeight(y2 - y1 + 1);

        if(getLoss) {
            const Matrix & lossMap = lossMaps(bestChild)[info->l_];
            info->loss_ = lossMap(info->y_, info->x_);
        } else {
            info->loss_ = 0.0F;
        }
    }

    // get scale prior and offset feature for toNode
    if(bestChild->scaleprior() != NULL) {
        Scaleprior::Param w = scalepriorFeatures_.col(info->l_);
        int idxPrior = pt.addScaleprior(w);
        toNode->getIdx()[PtNode::IDX_SCALEPRIOR] = idxPrior;
    }

    if(bestChild->bias() != NULL) {
        int idxBias = pt.AddBias(grammar_->featureBias());
        toNode->getIdx()[PtNode::IDX_BIAS] = idxBias;
    }

    // pass the parse info. to the best child
    int idxInfo = pt.addParseInfo(*info);//ptNode->idx()[PtNode::IDX_PARSEINFO];
    toNode->getIdx()[PtNode::IDX_PARSEINFO] = idxInfo;

    return true;
}

template<int Dimension>
bool DPInference::parseANDNode(int idx, vector<Node *> & gBFS,
                               vector<int> & ptBFS,
                               const FeaturePyr & pyramid, ParseTree & pt) {
    Node * gNode = gBFS[idx];
    if(gNode->type() != AND_NODE || !gNode->onOff()) {
        RGM_LOG(error, "Not an And-node or And-node is turned off.");
        return false;
    }

    int fromIdx = ptBFS[idx];
    PtNode * ptNode = pt.getNodeSet()[fromIdx];
    if(ptNode->getIdx()[PtNode::IDX_PARSEINFO] == -1) {
        RGM_LOG(error, "Need parse info. for the current pt node.");
        return false;
    }

    ParseInfo * info = pt.getParseInfo(ptNode);

    vector<Edge *> & outEdges(gNode->getOutEdges());

    if(outEdges.size() == 1 && outEdges[0]->type() == TERMINATION) {
        // Add an edge and a node to pt
        int idxG = grammar_->idxNode(outEdges[0]->getToNode());
        int t = static_cast<int>(outEdges[0]->getToNode()->type());
        int toIdx = pt.addNode(idxG, t);
        PtNode * toNode = pt.getNodeSet()[toIdx];

        idxG = grammar_->idxEdge(outEdges[0]);
        int edge = pt.addEdge(fromIdx, toIdx, idxG, outEdges[0]->type());

        int idxInfo = pt.addParseInfo(*info); //ptNode->idx()[PtNode::IDX_PARSEINFO];
        toNode->getIdx()[PtNode::IDX_PARSEINFO] = idxInfo;

        // Add the node to BFS
        gBFS.push_back(outEdges[0]->getToNode());
        ptBFS.push_back(toIdx);

        return true;
    }

    if(outEdges.size() == 1 && outEdges[0]->type() == DEFORMATION) {
        const MatrixXi & Ix = getDeformationX(gNode)[info->l_];
        const MatrixXi & Iy = getDeformationY(gNode)[info->l_];

        const int vpadx = VirtualPadding(pyramid.padx(), info->ds_);
        const int vpady = VirtualPadding(pyramid.pady(), info->ds_);

        // Location of ptNode without virtual padding
        int nvpX = info->x_ - vpadx;
        int nvpY = info->y_ - vpady;

        // Computing the toNode's location:
        //  - the toNode is (possibly) deformed to some other location
        //  - lookup its displaced location using the distance transform's
        //    argmax tables Ix and Iy
        int defX = Ix(nvpY, nvpX);
        int defY = Iy(nvpY, nvpX);

        // with virtual padding
        int toX = defX + vpadx;
        int toY = defY + vpady;

        // get deformation vectors
        int dx = info->x_ - toX;
        int dy = info->y_ - toY;

        if(ptNode->idx()[PtNode::IDX_DEF] != -1) {
            RGM_LOG(error, "Parsing wrong deformation AND-node");
            return false;
        }

        if(ptNode->idx()[PtNode::IDX_DEF] == -1)
            ptNode->getIdx()[PtNode::IDX_DEF] =
                pt.addDeformation(dx, dy,
                                  gNode->isLRFlip() &&
                                  grammar_->isLRFlip() &&
                                  grammar_->sharedLRFlip());

        // Look up the score of toNode
        const Matrix & score = scoreMaps(outEdges[0]->toNode())[info->l_];
        Scalar s = score(defY, defX);

        // Add an edge and a node to pt
        int idxG = grammar_->idxNode(outEdges[0]->getToNode());
        int t = static_cast<int>(outEdges[0]->getToNode()->type());
        int toIdx = pt.addNode(idxG, t);
        PtNode * toNode = pt.getNodeSet()[toIdx];

        idxG = grammar_->idxEdge(outEdges[0]);
        int edge = pt.addEdge(fromIdx, toIdx, idxG, outEdges[0]->type());

        // Detection scale and window
        Scalar scale = static_cast<Scalar>(grammar_->cellSize()) /
                       pyramid.scales()[info->l_];
        const Rectangle & detectWind = outEdges[0]->toNode()->detectWindow();

        // compute and record image coordinates of the detection window
        Scalar x1 = (toX - pyramid.padx() * std::pow<int>(2, info->ds_)) * scale;
        Scalar y1 = (toY - pyramid.pady() * std::pow<int>(2, info->ds_)) * scale;
        Scalar x2 = x1 + detectWind.width() * scale - 1;
        Scalar y2 = y1 + detectWind.height() * scale - 1;

        ParseInfo pinfo(0, info->l_, toX, toY, info->ds_, dx, dy, s,
                        Rectangle_<Scalar>(x1, y1, x2 - x1 + 1, y2 - y1 + 1));

        toNode->getIdx()[PtNode::IDX_PARSEINFO] = pt.addParseInfo(pinfo);

        // Add the node to BFS
        gBFS.push_back(outEdges[0]->getToNode());
        ptBFS.push_back(toIdx);

        return true;
    }

    RGM_CHECK(outEdges.size() >= 1 && outEdges[0]->type() == COMPOSITION, error);

    for(int i = 0; i < outEdges.size(); ++i) {
        if(!outEdges[i]->onOff())
            continue;

        // get anchor
        const Anchor & anchor = outEdges[i]->toNode()->anchor();
        int ax = anchor(0);
        int ay = anchor(1);
        int ads = anchor(2);

        // compute the location of toNode
        int toX = info->x_ * std::pow<int>(2, ads) + ax;
        int toY = info->y_ * std::pow<int>(2, ads) + ay;
        int toL = info->l_ - grammar_->interval() * ads;

        // Accumulate rescalings relative to ptNode
        int tods = info->ds_ + ads;

        // get the score of toNode
        const Matrix & score = scoreMaps(outEdges[i]->toNode())[toL];
        int nvpX = toX - VirtualPadding(pyramid.padx(), tods);
        int nvpY = toY - VirtualPadding(pyramid.pady(), tods);
        Scalar s = score(nvpY, nvpX);

        // Detection scale and window
        Scalar scale = static_cast<Scalar>(grammar_->cellSize()) /
                       pyramid.scales()[toL];
        const Rectangle & detectWind = outEdges[i]->toNode()->detectWindow();

        // compute and record image coordinates of the detection window
        Scalar x1 = (toX - pyramid.padx() * std::pow<int>(2, tods)) * scale;
        Scalar y1 = (toY - pyramid.pady() * std::pow<int>(2, tods)) * scale;
        Scalar x2 = x1 + detectWind.width() * scale - 1;
        Scalar y2 = y1 + detectWind.height() * scale - 1;

        // Add an edge and a node to pt
        int idxG = grammar_->idxNode(outEdges[i]->getToNode());
        int t = static_cast<int>(outEdges[i]->getToNode()->type());
        int toIdx = pt.addNode(idxG, t);
        PtNode * toNode = pt.getNodeSet()[toIdx];

        idxG = grammar_->idxEdge(outEdges[i]);
        int edge = pt.addEdge(fromIdx, toIdx, idxG, outEdges[i]->type());

        ParseInfo pinfo(0, toL, toX, toY, tods, 0, 0, s,
                        Rectangle_<Scalar>(x1, y1, x2 - x1 + 1, y2 - y1 + 1));

        toNode->getIdx()[PtNode::IDX_PARSEINFO] = pt.addParseInfo(pinfo);

        // Add the node to BFS
        gBFS.push_back(outEdges[i]->getToNode());
        ptBFS.push_back(toIdx);
    }         // for i

    return true;
}

template<int Dimension>
bool DPInference::parseTNode(int idx, vector<Node *> & gBFS,
                             vector<int> & ptBFS,
                             const FeaturePyr & pyramid, ParseTree & pt) {
    Node * gNode = gBFS[idx];
    if(gNode->type() != T_NODE || !gNode->onOff()) {
        RGM_LOG(error, "Not an T-node or T-node is turned off");
        return false;
    }

    int fromIdx = ptBFS[idx];
    PtNode * ptNode = pt.getNodeSet()[fromIdx];
    if(ptNode->getIdx()[PtNode::IDX_PARSEINFO] == -1) {
        RGM_LOG(error, "Need parse info. for the current pt node.");
        return false;
    }

    if(param_->createSample_) {
        ParseInfo * info = pt.getParseInfo(ptNode);

        int ds2 = std::pow<int>(2, info->ds_) - 1;
        int fy = info->y_ - pyramid.pady() * ds2;
        int fx = info->x_ - pyramid.padx() * ds2;
        int wd = gNode->appearance()->w().cols();
        int ht = gNode->appearance()->w().rows();
        AppParam w = pyramid.levels()[info->l_].block(fy, fx, ht, wd);

        //int bs = 20;
        /*cv::Mat img = FeaturePyramid::visualize(pyramid.levels()[info->l_], bs);
         * cv::rectangle(img, cv::Rect(fx*bs, fy*bs, gNode->appearance()->w().cols()*bs, gNode->appearance()->w().rows()*bs), cv::Scalar::all(255), 3);
         * cv::imshow("HOG", img);
         * cv::waitKey(0);
         *
         * if ( !getTestImg().empty() ) {
         *  cv::rectangle(getTestImg(), info->cvRect(), cv::Scalar::all(0), 2);
         *  cv::imshow("create sample", getTestImg());
         *  cv::waitKey(0);
         * }*/

        //FeaturePyramid::visualize(w, bs);

        ptNode->getIdx()[PtNode::IDX_APP] =
            pt.addAppearance(w, grammar_->featType(), gNode->isLRFlip() &&
                             grammar_->isLRFlip() &&
                             grammar_->sharedLRFlip());

        if(pt.appUsage().size() == 0)
            pt.getAppUsage().assign(grammar_->appearanceSet().size(), 0);

        pt.getAppUsage()[grammar_->idxAppearance(gNode->appearance())] += 1;
    }

    if(param_->createRootSample2x_) {
        ParseInfo * info = pt.getParseInfo(ptNode);

        int ads = 1;
        int ax = 0;
        int ay = 0;

        int ads2 = std::pow<int>(2, ads);

        int toX = info->x_ * ads2 + ax;
        int toY = info->y_ * ads2 + ay;
        int toL = info->l_ - grammar_->interval() * ads;

        int ds2 = ads2 - 1;
        int fy = toY - pyramid.pady() * ds2;
        int fx = toX - pyramid.padx() * ds2;

        int wd = gNode->appearance()->w().cols() * ads2;
        int ht = gNode->appearance()->w().rows() * ads2;

        assert(pt.appearaceX() == NULL);
        pt.getAppearaceX() = new AppParam();

        if((fy >= 0) && (fy + ht <= pyramid.levels()[toL].rows()) &&
                (fx >= 0) && (fx + wd <= pyramid.levels()[toL].cols()))
            *pt.getAppearaceX() = pyramid.levels()[toL].block(fy, fx, ht, wd);
        else {
            *pt.getAppearaceX() =
                Appearance::Param::Constant(
                    ht, wd, pyramid.levels()[toL](0, 0));

            int x1 = std::max<int>(fx, 0);
            int x2 = std::min<int>(fx + wd, pyramid.levels()[toL].cols());
            int y1 = std::max<int>(fy, 0);
            int y2 = std::min<int>(fy + ht, pyramid.levels()[toL].rows());
            int wd1 = x2 - x1;
            int ht1 = y2 - y1;

            int fx2 = (fx >= 0) ? 0 : -fx;
            int fy2 = (fy >= 0) ? 0 : -fy;

            pt.getAppearaceX()->block(fy2, fx2, ht1, wd1) =
                pyramid.levels()[toL].block(y1, x1, ht1, wd1);
        }
        if(gNode->isLRFlip() && grammar_->isLRFlip() &&
                grammar_->sharedLRFlip())
            *pt.getAppearaceX() = FeaturePyr::Flip(
                                      *pt.getAppearaceX(), grammar_->featType());

        if(pt.appUsage().size() == 0)
            pt.getAppUsage().assign(grammar_->appearanceSet().size(), 0);

        pt.getAppUsage()[grammar_->idxAppearance(gNode->appearance())] += 1;

        //int bs = 20;
        //FeaturePyramid::visualize(*pt.appearaceX(), bs);

        //Appearance::Param w = pyramid.levels()[info->l_].block(info->y_, info->x_, gNode->appearance()->w().rows(), gNode->appearance()->w().cols());
        //FeaturePyramid::visualize(w, bs);
    }

    if(param_->computeTNodeScores_) {
        ParseInfo * info = pt.getParseInfo(ptNode);

        int ds2 = std::pow<int>(2, info->ds_) - 1;
        int fy = info->y_ - pyramid.pady() * ds2;
        int fx = info->x_ - pyramid.padx() * ds2;

            const typename FeaturePyr::Level feat =
                pyramid.levels()[info->l_].block(
                    fy, fx, gNode->appearance()->w().rows(),
                    gNode->appearance()->w().cols());

            Eigen::Map<const Matrix, Eigen::Aligned> mapF(
                feat.data()->data(),
                gNode->appearance()->w().rows(),
                gNode->appearance()->w().cols() * Dimension);

            AppParam w = gNode->appearanceParam();
            Eigen::Map<const Matrix, Eigen::Aligned> mapW(
                w.data()->data(), w.rows(), w.cols() * Dimension);

            info->score_ = (mapF.cwiseProduct(mapW)).sum();

//                        const Matrix & score = scoreMaps(gNode)[info->l_];
//                        Scalar tmp = score(info->y_, info->x_);

//                        if ( std::abs(tmp - info->score_) > 1e-5) {
//                            int debug = 1;
//                        }

    }

    return true;
}

template<int Dimension>
const vector<Matrix> & DPInference::scoreMaps(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    if(n->type() == OR_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == SWITCHING)

        return scoreMaps(n->outEdges()[0]->toNode());

    if(n->type() == AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == TERMINATION)

        return scoreMaps(n->outEdges()[0]->toNode());

    Maps::const_iterator iter = scoreMaps_.find(n->tag());
    //    if ( iter == scoreMaps_.end() ) {
    //        int debug = 1;
    //    }
    RGM_CHECK_NOTEQ(iter, scoreMaps_.end());

    return iter->second;
}

template<int Dimension>
vector<Matrix> & DPInference::getScoreMaps(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    if(n->type() == OR_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == SWITCHING)

        return getScoreMaps(n->outEdges()[0]->toNode());

    if(n->type() == AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == TERMINATION)

        return getScoreMaps(n->outEdges()[0]->toNode());

    Maps::iterator iter = scoreMaps_.find(n->tag());
    if(iter == scoreMaps_.end())
        scoreMaps_.insert(std::make_pair(n->tag(), vector<Matrix>()));

    return scoreMaps_[n->tag()];
}

template<int Dimension>
void DPInference::setScoreMaps(const Node * n, int nbLevels, vector<Matrix> & s,
                               const vector<bool> & validLevels) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    if(n->type() != T_NODE)
        return;

    vector<Matrix> & m(getScoreMaps(n));

    m.resize(nbLevels);

    for(int i = 0, j = 0; i < nbLevels; ++i) {
        if(validLevels[i]) {
            m[i].swap(s[j]);
            ++j;
        } else
            m[i] = Matrix::Constant(1, 1, -Inf);
    }
}

template<int Dimension>
const vector<Matrix> & DPInference::scoreMapCopies(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    if(n->type() == OR_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == SWITCHING)

        return scoreMapCopies(n->outEdges()[0]->toNode());

    if(n->type() == AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == TERMINATION)

        return scoreMapCopies(n->outEdges()[0]->toNode());

    Maps::const_iterator iter = scoreMapCopies_.find(n->tag());
    RGM_CHECK_NOTEQ(iter, scoreMapCopies_.end());

    return iter->second;
}

template<int Dimension>
vector<Matrix> & DPInference::getScoreMapCopies(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    if(n->type() == OR_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == SWITCHING)

        return getScoreMapCopies(n->outEdges()[0]->toNode());

    if(n->type() == AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == TERMINATION)

        return getScoreMapCopies(n->outEdges()[0]->toNode());

    Maps::iterator iter = scoreMapCopies_.find(n->tag());
    if(iter == scoreMapCopies_.end())
        scoreMapCopies_.insert(std::make_pair(n->tag(), vector<Matrix>()));

    return scoreMapCopies_[n->tag()];
}

template<int Dimension>
const vector<bool> & DPInference::scoreMapStatus(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    if(n->type() == OR_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == SWITCHING)

        return scoreMapStatus(n->outEdges()[0]->toNode());

    if(n->type() == AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == TERMINATION)

        return scoreMapStatus(n->outEdges()[0]->toNode());

    Status::const_iterator iter = scoreMapStatus_.find(n->tag());
    RGM_CHECK_NOTEQ(iter, scoreMapStatus_.end());

    return iter->second;
}

template<int Dimension>
vector<bool> & DPInference::getScoreMapStatus(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    if(n->type() == OR_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == SWITCHING)

        return getScoreMapStatus(n->outEdges()[0]->toNode());

    if(n->type() == AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == TERMINATION)

        return getScoreMapStatus(n->outEdges()[0]->toNode());

    Status::iterator iter = scoreMapStatus_.find(n->tag());
    if(iter == scoreMapStatus_.end())
        scoreMapStatus_.insert(std::make_pair(n->tag(), vector<bool>()));

    return scoreMapStatus_[n->tag()];
}

template<int Dimension>
void DPInference::setScoreMapStatus(const Node * n, int l) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);
    int num = scoreMaps(n).size();

    RGM_CHECK_GE(l, 0);
    RGM_CHECK_LT(l, num);

    vector<bool> & status(getScoreMapStatus(n));
    if(num != status.size())
        status.assign(num, false);

    status[l] = true;
}

template<int Dimension>
void DPInference::copyScoreMaps(const Node * n) {
    if(!n->onOff())
        return;

    const vector<Matrix> & s(scoreMaps(n));

    vector<Matrix> & sCopies(getScoreMapCopies(n));
    sCopies.resize(s.size());

    std::copy(s.begin(), s.end(), sCopies.begin());

    // set score map status
    vector<bool> & status(getScoreMapStatus(n));
    status.assign(s.size(), false);
}

template<int Dimension>
void DPInference::recoverScoreMaps(const Node * n) {
    if(!n->onOff())
        return;

    const vector<bool> & status(scoreMapStatus(n));

    for(int i = 0; i < status.size(); ++i)
        if(status[i]) {
            getScoreMaps(n)[i] = scoreMapCopies(n)[i];
            getScoreMapStatus(n)[i] = false;
        }
}

template<int Dimension>
vector<MatrixXi> & DPInference::getDeformationX(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    ArgMaps::iterator iter = deformationX_.find(n->tag());
    if(iter == deformationX_.end())
        deformationX_.insert(std::make_pair(n->tag(), vector<MatrixXi>()));

    return deformationX_[n->tag()];
}

template<int Dimension>
vector<MatrixXi> & DPInference::getDeformationY(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    ArgMaps::iterator iter = deformationY_.find(n->tag());
    if(iter == deformationY_.end())
        deformationY_.insert(std::make_pair(n->tag(), vector<MatrixXi>()));

    return deformationY_[n->tag()];
}

template<int Dimension>
const vector<Matrix> & DPInference::lossMaps(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    if(n->type() == OR_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == SWITCHING)

        return lossMaps(n->outEdges()[0]->toNode());

    if(n->type() == AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == TERMINATION)

        return lossMaps(n->outEdges()[0]->toNode());

    Maps::const_iterator iter = lossMaps_.find(n->tag());

    if(iter != lossMaps_.end())
        return iter->second;
    else
        return vector<Matrix>();
}

template<int Dimension>
vector<Matrix> & DPInference::getLossMaps(const Node * n) {
    RGM_CHECK_NOTNULL(n);
    RGM_CHECK(n->onOff(), error);

    if(n->type() == OR_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == SWITCHING)

        return getLossMaps(n->outEdges()[0]->toNode());

    if(n->type() == AND_NODE &&
            n->outEdges().size() == 1 &&
            n->outEdges()[0]->type() == TERMINATION)

        return getLossMaps(n->outEdges()[0]->toNode());

    Maps::iterator iter = lossMaps_.find(n->tag());
    if(iter == lossMaps_.end())
        lossMaps_.insert(std::make_pair(n->tag(), vector<Matrix>()));

    return lossMaps_[n->tag()];
}

template<int Dimension>
vector<bool> DPInference::computeOverlapMaps(const vector<Rectangle> & bboxes,
                                             const FeaturePyr & pyr,
                                             OverlapMaps & overlapMaps,
                                             Scalar overlapThr) {
    vector<Node *> subcategory =
        grammar_->getSubcategoryRootAndNodes(true, true);

    const vector<Matrix> & s(scoreMaps(grammar_->rootNode()));

    // for each level
    overlapMaps.resize(s.size());
    vector<bool> valid(s.size(), false);

    for(int l = 0; l < s.size(); ++l) {
        // for each gt box
        overlapMaps[l].resize(bboxes.size());

        const int rows = s[l].rows();
        const int cols = s[l].cols();
        const Scalar scale = static_cast<Scalar>(grammar_->cellSize()) /
                             pyr.scales()[l];

        for(int b = 0; b < bboxes.size(); ++b) {
            // for each model
            overlapMaps[l][b].resize(subcategory.size());

            // at original image resolution
            Rectangle_<float> refBox(bboxes[b].x(), bboxes[b].y(),
                                     bboxes[b].width(), bboxes[b].height());
            Intersector_<float> inter(refBox, overlapThr, true);

            const bool imgClip = (bboxes[b].area() /
                                  float(pyr.imgWd() * pyr.imgHt())) < 0.7F;

            for(int c = 0; c < subcategory.size(); ++c) {
                const Rectangle & detWind = subcategory[c]->detectWindow();
                const float wd = detWind.width() * scale;
                const float ht = detWind.height() * scale;
                assert(wd > 0 && ht > 0);

                Matrix & o = overlapMaps[l][b][c];
                o = Matrix::Zero(rows, cols);

                for(int y = 0;
                        y < max<int>(o.rows(), pyr.levels()[l].rows());
                        ++y) {
                    float y1 = (y - pyr.pady()) * scale;
                    float y2 = y1 + ht - 1;
                    if(imgClip) {
                        y1 = min<float>(pyr.imgHt() - 1, max<float>(y1, 0));
                        y2 = min<float>(pyr.imgHt() - 1, max<float>(y2, 0));
                    }

                    for(int x = 0;
                            x < std::max<int>(o.cols(), pyr.levels()[l].cols());
                            ++x) {
                        float x1 = (x - pyr.padx()) * scale;
                        float x2 = x1 + wd - 1;
                        if(imgClip) {
                            x1 =
                                min<float>(pyr.imgWd() - 1,
                                           max<float>(x1, 0));
                            x2 =
                                min<float>(pyr.imgWd() - 1,
                                           max<float>(x2, 0));
                        }

                        Rectangle_<float> box(x1,
                                              y1,
                                              x2 - x1 + 1,
                                              y2 - y1 + 1);
                        float ov = 0;

                        if(inter(box,
                                 &ov) && (y < pyr.levels()[l].rows()) &&
                                (x < pyr.levels()[l].cols())) {
                            valid[l] = true;
                            // assumes that models only have one level of parts
                            if(l - pyr.interval() >= 0)
                                valid[l - pyr.interval()] = true;
                        }

                        if(y < o.rows() && x < o.cols())
                            o(y, x) = ov;
                    }         // for x
                }         // for y
            }         // for c
        }         // for b
    }         // for l

    for(int l = 0; l < s.size(); ++l)
        if(!valid[l])
            overlapMaps[l].clear();

    return valid;
}

template<int Dimension>
void DPInference::inhibitOutput(int idxBox,
                                OverlapMaps & overlapMap,
                                Scalar overlapThr, bool needCopy) {
    vector<Node *> subcategory =
        grammar_->getSubcategoryRootAndNodes(true, true);

    for(int c = 0; c < subcategory.size(); ++c) {
        Node * objComp = subcategory[c];

        vector<Matrix> & s(getScoreMaps(objComp));

        for(int i = 0; i < s.size(); ++i) {
            if(overlapMap[i].size() == 0)
                continue;

            if(needCopy)
                setScoreMapStatus(objComp, i);

            Matrix & o = overlapMap[i][idxBox][c];
            s[i] = (o.array() < overlapThr).select(-Inf, s[i]);
        }
    }         // for c

    computeORNode(grammar_->getRootNode());
}

template<int Dimension>
void DPInference::inhibitAllFg(OverlapMaps & overlapMap,
                               Scalar overlapThr, bool needCopy) {
    vector<Node *> subcategory =
        grammar_->getSubcategoryRootAndNodes(true, true);

    for(int c = 0; c < subcategory.size(); ++c) {
        Node * objComp = subcategory[c];

        vector<Matrix> & s(getScoreMaps(objComp));

        for(int i = 0; i < s.size(); ++i) {
            if(overlapMap[i].size() == 0)
                continue;

            if(needCopy)
                setScoreMapStatus(objComp, i);

            for(int j = 0; j < overlapMap[i].size(); ++j) {
                Matrix & o = overlapMap[i][j][c];
                s[i] = (o.array() >= overlapThr).select(-Inf, s[i]);
            }
        }
    }         // for c

    computeORNode(grammar_->getRootNode());
}

template<int Dimension>
void DPInference::inhibitAllFg(OverlapMaps & overlapMap,
                               Scalar minOvThr, Scalar maxOvThr,
                               bool needCopy) {
    vector<Node *> subcategory =
        grammar_->getSubcategoryRootAndNodes(true, true);

    for(int c = 0; c < subcategory.size(); ++c) {
        Node * objComp = subcategory[c];

        vector<Matrix> & s(getScoreMaps(objComp));

        for(int i = 0; i < s.size(); ++i) {
            if(overlapMap[i].size() == 0)
                continue;

            if(needCopy)
                setScoreMapStatus(objComp, i);

            for(int j = 0; j < overlapMap[i].size(); ++j) {
                Matrix & o = overlapMap[i][j][c];
                s[i] = (o.array() >= maxOvThr).select(-Inf, s[i]);
                //s[i] = (o.array() < minOvThr).select(-Inf, s[i]);
            }
        }
    }         // for c

    computeORNode(grammar_->getRootNode());
}

template<int Dimension>
void DPInference::applyLossAdjustment(int idxBox, int nbBoxes,
                                      OverlapMaps & overlapMap, Scalar fgOverlap,
                                      Scalar bgOverlap, bool needCopy) {
    vector<Node *> subcategory =
        grammar_->getSubcategoryRootAndNodes(true, true);

    for(int c = 0; c < subcategory.size(); ++c) {
        Node * objComp = subcategory[c];

        vector<Matrix> & s(getScoreMaps(objComp));

        vector<Matrix> & lmap(getLossMaps(objComp));
        lmap.resize(s.size(), Matrix::Zero(1, 1));

        for(int l = 0; l < s.size(); ++l) {
            if(overlapMap[l].size() == 0)
                continue;

            Matrix & o = overlapMap[l][idxBox][c];
            Matrix & loss = lmap[l];

            loss = Matrix::Zero(o.rows(), o.cols());

            // PASCAL VOC loss
            loss = (o.array() < 0.5F).select(1, loss);

            // fg overlap
            // Require at least some overlap with the foreground bounding box
            //    In an image with multiple objects, this constraint encourages a
            //    diverse set of false positives (otherwise, they will tend to come
            //    from the same high-scoring / low-overlapping region of the image
            //    -- i.e. somewhere in the background)
            loss = (o.array() < fgOverlap).select(-Inf, loss);

            // bg overlap
            // Mark root locations that have too much overlap with background boxes as invalid
            //     We don't want to select detections of other foreground objects
            //     in the image as false positives (i.e., no true positive should
            //     be allowed to be used as a false positive)
            for(int i = 0; i < nbBoxes; ++i) {
                if(i == idxBox)
                    continue;

                Matrix & obg = overlapMap[l][i][c];

                loss = (obg.array() >= bgOverlap).select(-Inf, loss);
            }

            // loss adjustment
            s[l] += loss;

            if(needCopy)
                setScoreMapStatus(objComp, l);
        }         // for l
    }         // for c

    computeORNode(grammar_->getRootNode());
}

template<int Dimension>
void DPInference::visScoreMaps(std::string & saveDir, ParseTree * pt) {
    const vector<Node *> &bfs = grammar_->nodeBFS();

    std::string saveName;
    std::map<int, cv::Mat> maps;
    for ( int i = 0; i < bfs.size(); ++i ) {
        vector<Matrix> & scoremaps = getScoreMaps(bfs[i]);
        vector<Mat> cmap = OpencvUtil::showColorMap(scoremaps);
        int l = -1;
        if ( pt ) {
            for(int n = 0; n < pt->nodeSet().size(); ++n) {
                const PtNode * ptnode = pt->nodeSet()[n];
                const ParseInfo * info = pt->parseInfo(ptnode);
                const Node * gnode =
                    grammar_->findNode(ptnode->idx()[PtNode::IDX_G]);
                if ( gnode == bfs[i] ) {
                    l = info->l_;
                    Mat map = cmap[l].clone();
                    maps.insert(std::make_pair(ptnode->idx()[PtNode::IDX_MYSELF], map));
                    break;
                }
            }
        }
        if ( l != -1 ) {
            cv::rectangle(cmap[l], cv::Rect(0, 0, cmap[l].cols, cmap[l].rows),
                          cv::Scalar(0, 0, 255), 3);
        }
        for ( int j = 0; j < cmap.size(); ++j ) {
            saveName = saveDir + NumToString_<int>(i, 3) + "_" +
                    NumToString_<int>(j, 3) + ".png";
            cv::imwrite(saveName, cmap[j]);
        }
    }

    if (pt) {
        string mapName;
        for(int i = 0; i < pt->nodeSet().size(); ++i) {
            const PtNode * ptnode = pt->nodeSet()[i];
            int idx = ptnode->idx()[PtNode::IDX_MYSELF];
            cv::Mat map = maps[idx];

            mapName = saveDir + NumToString_<int>(idx, 5) + ".png";
            cv::imwrite(mapName, map);
        }
        saveName = saveDir + "pt.png";
        pt->visualize(saveName);
    }
}

template<int Dimension>
void DPInference::getPtScoreMaps(ParseTree & pt,
                                 std::map<int, cv::Mat> & maps,
                                 int cropSz, string saveName,
                                 std::map<int, Scalar> * fom) {
    RGM_CHECK_EQ(pt.grammar(), grammar_);
    maps.clear();

    int tnodeIdx = 0;
    for(int i = 0; i < pt.nodeSet().size(); ++i) {
        const PtNode * ptnode = pt.nodeSet()[i];
        const ParseInfo * info = pt.parseInfo(ptnode);
        const Node * gnode =
            grammar_->findNode(ptnode->idx()[PtNode::IDX_G]);
        vector<Matrix> & scoremaps = getScoreMaps(gnode);
        vector<Matrix> smap(1, scoremaps[info->l_]);
        vector<cv::Mat> cmap = OpencvUtil::showColorMap(smap);
        int x = std::max<int>(0, info->x_ - cropSz);
        int y = std::max<int>(0, info->y_ - cropSz);
        int wd = std::min<int>(cmap[0].cols, info->x_ + cropSz) - x;
        int ht = std::min<int>(cmap[0].rows, info->y_ + cropSz) - y;

        cv::Mat bigmap;
        cv::Size sz(std::max(100, wd), std::max(100, ht));
        cv::resize(cmap[0](cv::Rect(x, y, wd, ht)), bigmap, sz);

        if(gnode->type() == T_NODE && tnodeIdx < rgbTableSz) {
            cv::rectangle(bigmap, cv::Rect(1, 1, bigmap.cols - 1, bigmap.rows - 1),
                          rgbTable[tnodeIdx], 3);
            tnodeIdx++;
        }

        maps.insert(std::make_pair(ptnode->idx()[PtNode::IDX_G], bigmap));
    }

    if(!saveName.empty()) {
        string strDir = FileUtil::GetParentDir(saveName);
        FileUtil::VerifyDirectoryExists(strDir, true);

        string strExt(".png");
        string mapName;
        for(int i = 0; i < pt.nodeSet().size(); ++i) {
            const PtNode * ptnode = pt.nodeSet()[i];
            int idx = ptnode->idx()[PtNode::IDX_G];
            cv::Mat map = maps[idx];

            mapName = strDir + NumToString_<int>(idx, 5) + strExt;
            cv::imwrite(mapName, map);
        }

        if(fom == NULL) {
            std::map<int, Scalar> fom1;
            computeIntrackability(pt, fom1);
            pt.visualize(saveName, &fom1);
        } else
            pt.visualize(saveName, fom);
    }
}

template<int Dimension>
vector<int> DPInference::computeIntrackability(
    ParseTree & pt,
    std::map<int, Scalar> & out, Scalar thr) {
    RGM_CHECK_EQ(pt.grammar(), grammar_);

    out.clear();

    vector<int> turnedOffTnodes;

    for(int i = 0; i < pt.nodeSet().size(); ++i) {
        PtNode * ptnode = pt.getNodeSet()[i];
        ParseInfo * info = pt.getParseInfo(ptnode);
        const Node * gnode =
            grammar_->findNode(ptnode->idx()[PtNode::IDX_G]);
        Matrix scoremap = getScoreMaps(gnode)[info->l_];

        //        Matrix m =  (scoremap.array() == Inf).select(-Inf, scoremap);
        //        int num = (m.array() == -Inf).count();
        //        int num1 = scoremap.cols()*scoremap.rows() - num;

        //        Scalar maxcoeff = m.maxCoeff();

        //        m =  (m.array() == -Inf).select(Inf, m);
        //        Scalar mincoeff = m.minCoeff();

        //        Matrix m1 = (m.array() == Inf).select(mincoeff, m);
        //        m1 = (m1.array() - mincoeff) / (maxcoeff - mincoeff);

        //        //m1.array() /= m1.sum();

        //        Matrix m2 = (m.array() == Inf).select(1, m1);
        //        m2 = (m2.array() <= 0).select(1, m2);

        //        m2 = m2.array().log();

        //        Scalar quality = -m1.cwiseProduct(m2).sum() / num1;

        //        Scalar s = scoremap.maxCoeff(); // scoremap(info->y_, info->x_);

        //        Matrix m =  (scoremap.array() == Inf).select(-Inf, scoremap);
        //        int num = (m.array() == -Inf).count();
        //        int num1 = scoremap.cols()*scoremap.rows() - num;

        //        Matrix m1 = (m.array() == -Inf).select(s, m);

        //        m1 = s - m1.array();

        //        Scalar quality = (m1.sum() / num1);

        //        Matrix m =  (scoremap.array() == Inf).select(-Inf, scoremap);
        //        int num = (m.array() == -Inf).count();
        //        int num1 = scoremap.cols()*scoremap.rows() - num;

        //        Matrix m1 = (m.array() == -Inf).select(0, m);
        //        Scalar mean = m1.sum() / num1;

        //        m1 = (m.array() == -Inf).select(mean, m);
        //        m1.array() -= mean;

        //        Scalar std = sqrt(m1.squaredNorm() / (num1 - 1));

        Matrix m = (scoremap.array() == Inf).select(-Inf, scoremap);
        int num = (m.array() == -Inf).count();
        int num1 = scoremap.cols() * scoremap.rows() - num;

        Matrix m1 = (m.array() == -Inf).select(0, m);
        Scalar mean = m1.sum() / num1;

        m1 = (m.array() == -Inf).select(mean, m);
        m1.array() -= mean;

        Scalar std = sqrt(m1.squaredNorm() / (num1 - 1));

        Scalar quality = std::abs(scoremap(info->y_, info->x_) - mean) / std;

        if(gnode->type() == T_NODE && quality < thr)
            turnedOffTnodes.push_back(grammar_->idxNode(gnode));

        out.insert(std::make_pair(ptnode->idx()[PtNode::IDX_MYSELF], quality));

        info->goodness_ = quality;
    }

    return turnedOffTnodes;
}

template<int Dimension>
Scalar DPInference::computeIntrackability(ParseTree & pt) {
    RGM_CHECK_EQ(pt.grammar(), grammar_);

    PtNode * ptnode = pt.getRootNode();

    ParseInfo * info = pt.getParseInfo(ptnode);
    const Node * gnode = grammar_->findNode(ptnode->idx()[PtNode::IDX_G]);
    Matrix scoremap = getScoreMaps(gnode)[info->l_];

    //        Matrix m =  (scoremap.array() == Inf).select(-Inf, scoremap);
    //        int num = (m.array() == -Inf).count();
    //        int num1 = scoremap.cols()*scoremap.rows() - num;

    //        Scalar maxcoeff = m.maxCoeff();

    //        m =  (m.array() == -Inf).select(Inf, m);
    //        Scalar mincoeff = m.minCoeff();

    //        Matrix m1 = (m.array() == Inf).select(mincoeff, m);
    //        m1 = (m1.array() - mincoeff) / (maxcoeff - mincoeff);

    //        //m1.array() /= m1.sum();

    //        Matrix m2 = (m.array() == Inf).select(1, m1);
    //        m2 = (m2.array() <= 0).select(1, m2);

    //        m2 = m2.array().log();

    //        Scalar quality = -m1.cwiseProduct(m2).sum() / num1;

    //        Scalar s = scoremap.maxCoeff(); // scoremap(info->y_, info->x_);

    //        Matrix m =  (scoremap.array() == Inf).select(-Inf, scoremap);
    //        int num = (m.array() == -Inf).count();
    //        int num1 = scoremap.cols()*scoremap.rows() - num;

    //        Matrix m1 = (m.array() == -Inf).select(s, m);

    //        m1 = s - m1.array();

    //        Scalar quality = (m1.sum() / num1);

    //        Matrix m =  (scoremap.array() == Inf).select(-Inf, scoremap);
    //        int num = (m.array() == -Inf).count();
    //        int num1 = scoremap.cols()*scoremap.rows() - num;

    //        Matrix m1 = (m.array() == -Inf).select(0, m);
    //        Scalar mean = m1.sum() / num1;

    //        m1 = (m.array() == -Inf).select(mean, m);
    //        m1.array() -= mean;

    //        Scalar std = sqrt(m1.squaredNorm() / (num1 - 1));

    Matrix m = (scoremap.array() == Inf).select(-Inf, scoremap);
    int num = (m.array() == -Inf).count();
    int num1 = scoremap.cols() * scoremap.rows() - num;

    Matrix m1 = (m.array() == -Inf).select(0, m);
    Scalar mean = m1.sum() / num1;

    m1 = (m.array() == -Inf).select(mean, m);
    m1.array() -= mean;

    Scalar std = sqrt(m1.squaredNorm() / (num1 - 1));

    Scalar quality = std::abs(scoremap(info->y_, info->x_) - mean) / std;


    info->goodness_ = quality;

    return quality;
}

template<int Dimension>
void DPInference::release() {
    scoreMaps_.clear();
    scoreMapCopies_.clear();
    scoreMapStatus_.clear();
    deformationX_.clear();
    deformationY_.clear();
    lossMaps_.clear();
}

/// Instantiation
INSTANTIATE_CLASS_(DPInference_);


} // namespace RGM
