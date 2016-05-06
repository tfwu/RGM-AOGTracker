/// This file is adapted from UCLA ImageParser by Brandon Rothrock (rothrock@cs.ucla.edu)

#ifndef RGM_XMLREADER_HPP_
#define RGM_XMLREADER_HPP_

#include <vector>
#include <string>
#include <string.h>
#include <fstream>

#include "RapidXML/rapidxml.hpp"

namespace RGM {

using std::vector;
using std::string;

/// Read xml file
class XMLData {
  public:
    struct XMLNode {
        XMLNode();
        XMLNode(char* _name, char* _value);

        ~XMLNode();

        XMLNode* Clone();

        void SetName(char* _name);
        void SetValue(char* _value);
        void SetChildren(vector<XMLNode*>& _children);
        void AddChild(XMLNode* pNode);

        char* _name;
        char* _value;
        int _childCount;
        XMLNode* _parent;
        XMLNode** _children;
    };

    XMLData();

    void Clear();

    void ReadFromFile(string fileName, string firstNodeName);
    void ReadFromString(char* str, string firstNodeName);
    void WriteToFile(string fileName);

    XMLNode* GetNode(string path);
    XMLNode* GetNode(string path, XMLNode* pNode);
    XMLNode* GetRootNode();

    XMLNode* FindFirst(string path, string childName, string childValue);
    XMLNode* FindFirst(string path, string childName, string field,
                       string query);
    XMLNode* FindFirst(string basePath, string childName, string field1,
                       string query1, string field2, string query2);

    vector<XMLNode *> GetNodes(const string& path);
    vector<XMLNode *> GetNodes(const string& path, XMLNode* pNode);

    string GetNodeName(XMLNode* pNode);
    string GetString(string path);
    int GetInt(string path);
    float GetFloat(string path);
    bool GetBoolean(string path);

    string GetString(XMLNode* pNode);
    int   GetInt(XMLNode* pNode);
    float GetFloat(XMLNode* pNode);
    bool  GetBoolean(XMLNode* pNode);

    string GetString(string path, XMLNode* pNode);
    int   GetInt(string path, XMLNode* pNode);
    float GetFloat(string path, XMLNode* pNode);
    bool  GetBoolean(string path, XMLNode* pNode);

    void RemoveNode(XMLNode* pNode);
    void InsertNode(XMLNode* pParent, XMLNode* pNode);

  private:
    XMLNode* ReadNode(rapidxml::xml_node<>* node);
    void     WriteNode(XMLNode* node, std::ofstream& ofs);

    XMLNode* _pRootNode;
};

} // namespace RGM

#endif // RGM_XMLREADER_HPP_

