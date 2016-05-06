#include <fstream>
#include <iostream>
#include <boost/lexical_cast.hpp>

#include "xml_reader.hpp"

namespace RGM
{
using namespace std;
using namespace rapidxml;

// -------- XMLData::XMLNode --------

XMLData::XMLNode::XMLNode() :
    _name(NULL), _value(NULL), _childCount(0), _children(NULL), _parent(NULL)
{
}

XMLData::XMLNode::XMLNode(char* _name, char* _value) :
    _name(NULL), _value(NULL), _childCount(0), _children(NULL), _parent(NULL)
{
    if (_name != NULL) {
        this->_name = new char[strlen(_name)+1];
        strcpy(this->_name, _name);
    }
    if (_value != NULL) {
        this->_value = new char[strlen(_value)+1];
        strcpy(this->_value, _value);
    }
}

XMLData::XMLNode::~XMLNode()
{
    if (_name != NULL) {
        delete[] _name;
    }
    if (_value != NULL) {
        delete[] _value;
    }
    for (int i=0; i<_childCount; i++) {
        delete _children[i];
    }
    if (_children != NULL) {
        delete[] _children;
    }
}

XMLData::XMLNode* XMLData::XMLNode::Clone()
{
    XMLNode* pCopy = new XMLNode(_name, _value);
    pCopy->_childCount = _childCount;
    pCopy->_parent = _parent;
    pCopy->_children = new XMLNode*[_childCount];
    for (int i=0; i<_childCount; i++) {
        pCopy->_children[i] = _children[i]->Clone();
    }
    return pCopy;
}

void XMLData::XMLNode::SetName(char* _name)
{
    if (this->_name != NULL) {
        delete[] this->_name;
        this->_name = NULL;
    }
    if (_name != NULL) {
        this->_name = new char[strlen(_name)+1];
        strcpy(this->_name, _name);
    }
}

void XMLData::XMLNode::SetValue(char* _value)
{
    if (this->_value != NULL) {
        delete[] this->_value;
        this->_value = NULL;
    }
    if (_value != NULL) {
        this->_value = new char[strlen(_value)+1];
        strcpy(this->_value, _value);
    }
}

void XMLData::XMLNode::SetChildren(vector<XMLNode*>& _children)
{
    this->_childCount = (int)_children.size();
    if (this->_childCount > 0) {
        this->_children = new XMLNode*[this->_childCount];
        for (int i=0; i<this->_childCount; i++) {
            this->_children[i] = _children[i];
            this->_children[i]->_parent = this;
        }
    }
}

void XMLData::XMLNode::AddChild(XMLNode* pNode)
{
    XMLNode** newChildren = new XMLNode*[_childCount+1];
    for (int i=0; i<_childCount; i++) {
        newChildren[i] = _children[i];
    }
    newChildren[_childCount] = pNode;
    if (_children != NULL) {
        delete[] _children;
    }
    _children = newChildren;
    _childCount++;
    pNode->_parent = this;
}


// -------- XMLData --------

XMLData::XMLData() :
    _pRootNode(NULL)
{
}

void XMLData::Clear()
{
    if (_pRootNode != NULL) {
        RemoveNode(_pRootNode);
        _pRootNode = NULL;
    }
}

string XMLData::GetString(string path)
{
    return GetString(GetNode(path, _pRootNode));
}

int XMLData::GetInt(string path)
{
    return GetInt(GetNode(path, _pRootNode));
}

float XMLData::GetFloat(string path)
{
    return GetFloat(GetNode(path, _pRootNode));
}

bool XMLData::GetBoolean(string path)
{
    return (GetInt(path) != 0);
}

string XMLData::GetString(XMLNode* pNode)
{
    if (pNode == NULL) {
        return string();
    }
    return string(pNode->_value);
}

int XMLData::GetInt(XMLNode* pNode)
{
    if (pNode == NULL) {
        return 0;
    }
    return boost::lexical_cast<int>(pNode->_value);
}

float XMLData::GetFloat(XMLNode* pNode)
{
    if (pNode == NULL) {
        return 0;
    }
    return boost::lexical_cast<float>(pNode->_value);
}

bool  XMLData::GetBoolean(XMLNode* pNode)
{
    return (GetInt(pNode) != 0);
}

string XMLData::GetString(string path, XMLNode* pNode)
{
    return GetString(GetNode(path, pNode));
}

int   XMLData::GetInt(string path, XMLNode* pNode)
{
    return GetInt(GetNode(path, pNode));
}

float XMLData::GetFloat(string path, XMLNode* pNode)
{
    return GetFloat(GetNode(path, pNode));
}

bool  XMLData::GetBoolean(string path, XMLNode* pNode)
{
    return (GetInt(path, pNode) != 0);
}

string XMLData::GetNodeName(XMLData::XMLNode* pNode)
{
    return string(pNode->_name);
}

XMLData::XMLNode* XMLData::GetNode(string path)
{
    return GetNode(path, _pRootNode);
}

XMLData::XMLNode* XMLData::GetNode(string path, XMLData::XMLNode* pNode)
{
    if (pNode == NULL) {
        return NULL;
    }
    string localPath = path;
    string remainingPath;
    bool leaf = true;
    int pathIndex = path.find_first_of("/");
    if (pathIndex != string::npos) {
        localPath = path.substr(0, pathIndex);
        remainingPath = path.substr(pathIndex+1, path.length()-pathIndex-1);
        leaf = false;
    }

    for (int i=0; i<pNode->_childCount; i++) {
        if (strcmp(pNode->_children[i]->_name, localPath.c_str()) == 0) {
            if (leaf) {
                return pNode->_children[i];
            } else {
                return GetNode(remainingPath, pNode->_children[i]);
            }
        }
    }

    return NULL;
}

XMLData::XMLNode* XMLData::GetRootNode()
{
    return _pRootNode;
}

vector<XMLData::XMLNode *> XMLData::GetNodes(const string& path)
{
    return GetNodes(path, _pRootNode);
}

vector<XMLData::XMLNode *> XMLData::GetNodes(const string& path, XMLNode* pNode)
{
    vector<XMLData::XMLNode *> nodes;

    if (pNode == NULL) {
        return nodes;
    }

    string localPath = path;
    string remainingPath;
    bool leaf = true;
    int pathIndex = path.find_first_of("/");
    if (pathIndex != string::npos) {
        localPath = path.substr(0, pathIndex);
        remainingPath = path.substr(pathIndex+1, path.length()-pathIndex-1);
        leaf = false;
    }

    for (int i=0; i<pNode->_childCount; i++) {
        if (strcmp(pNode->_children[i]->_name, localPath.c_str()) == 0) {
            if (leaf) {
                nodes.push_back( pNode->_children[i] );
            } else {
                vector<XMLData::XMLNode *> nodes1 =
                        GetNodes(remainingPath, pNode->_children[i]);
                for ( int j=0; j<nodes1.size(); ++j ) {
                    nodes.push_back( nodes1[j] );
                }
            }
        }
    }

    return nodes;
}

XMLData::XMLNode* XMLData::FindFirst(string path, string childName,
                                     string childValue)
{
    XMLNode* basenode = GetNode(path);
    if (basenode == NULL) {
        return NULL;
    }
    for (int i=0; i<basenode->_childCount; i++) {
        XMLNode* childnode = basenode->_children[i];
        string currentChildName = GetNodeName(childnode);
        if (currentChildName.compare(childName) != 0) {
            continue;
        }
        if (childValue.compare(childnode->_value) != 0) {
            continue;
        }
        return childnode;
    }
    return NULL;
}

XMLData::XMLNode* XMLData::FindFirst(string path, string childName,
                                     string field, string query)
{
    XMLNode* basenode = GetNode(path);
    if (basenode == NULL) {
        return NULL;
    }
    for (int i=0; i<basenode->_childCount; i++) {
        XMLNode* childnode = basenode->_children[i];
        string currentChildName = GetNodeName(childnode);
        if (currentChildName.compare(childName) != 0) {
            continue;
        }
        XMLNode* attrNode1 = GetNode(field, childnode);
        if (attrNode1 == NULL) {
            continue;
        }
        if (query.compare(attrNode1->_value) != 0) {
            continue;
        }
        return childnode;
    }
    return NULL;
}

XMLData::XMLNode* XMLData::FindFirst(string basePath, string childName,
                                     string field1, string query1,
                                     string field2, string query2)
{
    XMLNode* basenode = GetNode(basePath);
    if (basenode == NULL) {
        return NULL;
    }
    for (int i=0; i<basenode->_childCount; i++) {
        XMLNode* childnode = basenode->_children[i];
        string currentChildName = GetNodeName(childnode);
        if (currentChildName.compare(childName) != 0) {
            continue;
        }
        XMLNode* attrNode1 = GetNode(field1, childnode);
        if (attrNode1 == NULL) {
            continue;
        }
        if (query1.compare(attrNode1->_value) != 0) {
            continue;
        }
        XMLNode* attrNode2 = GetNode(field2, childnode);
        if (attrNode2 == NULL) {
            continue;
        }
        if (query2.compare(attrNode2->_value) != 0) {
            continue;
        }
        return childnode;
    }
    return NULL;
}

void XMLData::RemoveNode(XMLNode* pNode)
{
    if (pNode->_parent == NULL) {
        delete pNode;
        return;
    }

    int _childCount = pNode->_parent->_childCount;
    if (_childCount > 1) {
        XMLNode** _children = new XMLNode*[_childCount-1];
        int childIndex = 0;
        for (int i=0; i<_childCount; i++) {
            if (pNode->_parent->_children[i] == pNode) {
                continue;
            }
            _children[childIndex] = pNode->_parent->_children[i];
            childIndex++;
        }
        delete[] pNode->_parent->_children;
        pNode->_parent->_children = _children;
        pNode->_parent->_childCount--;
        delete pNode;
    } else {
        delete[] pNode->_parent->_children;
        pNode->_parent->_children = NULL;
        pNode->_parent->_childCount--;
        delete pNode;
    }
}

void XMLData::InsertNode(XMLNode* pParent, XMLNode* pNode)
{
    pParent->AddChild(pNode);
}


void XMLData::ReadFromFile(string fileName, string firstNodeName)
{
    Clear();

    ifstream ifs;
    ifs.open(fileName.c_str(), ios::in);
    string str((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
    char* str2 = new char[str.length()+1];
    strcpy(str2, str.c_str());
    ifs.close();

    ReadFromString(str2, firstNodeName);

    delete[] str2;
}

void XMLData::ReadFromString(char* str, string firstNodeName)
{
    Clear();
    // Note: this destructively reads from input string
    xml_document<> doc;
    doc.parse<parse_no_data_nodes | parse_trim_whitespace>((char*)str);
    xml_node<>* configNode = doc.first_node(firstNodeName.c_str());
    _pRootNode = ReadNode(configNode);
    doc.clear();
}

XMLData::XMLNode* XMLData::ReadNode(rapidxml::xml_node<>* node)
{

    if (node->type() != node_element) {
        return NULL;
    }

    XMLNode* pConfigNode = new XMLNode(node->name(), node->value());

    vector<XMLNode*> _children;
    for (xml_node<>* child = node->first_node(); child != NULL;
         child = child->next_sibling()) {
        XMLNode* pChildConfigNode = ReadNode(child);
        _children.push_back(pChildConfigNode);
    }

    for (xml_attribute<>* attr = node->first_attribute(); attr != NULL;
         attr = attr->next_attribute()) {
        XMLNode* pAttrNode = new XMLNode(attr->name(), attr->value());
        _children.push_back(pAttrNode);
    }

    pConfigNode->SetChildren(_children);
    return pConfigNode;
}

void XMLData::WriteToFile(string fileName)
{

    ofstream ofs;
    ofs.open(fileName.c_str(), ios::out);

    WriteNode(_pRootNode, ofs);

    ofs.close();
}

void XMLData::WriteNode(XMLNode* node, ofstream& ofs)
{
    ofs << "<" << node->_name << ">" << endl;
    if ((node->_value != NULL) && (strlen(node->_value) > 0)) {
        ofs << node->_value << endl;
    }

    for (int i=0; i<node->_childCount; i++) {
        WriteNode(node->_children[i], ofs);
    }

    ofs << "</" << node->_name << ">" << endl;
}

} // namespace RGM
