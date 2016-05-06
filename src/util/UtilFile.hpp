#ifndef RGM_FILEUTILITY_HPP_
#define RGM_FILEUTILITY_HPP_

#include "common.hpp"

namespace RGM {

using std::string;
using std::vector;

namespace fs = boost::filesystem;

/// Utility functions for file management
class FileUtil {

public:
	/// check if a file or dir exists
	static bool exists(const string & input);
	/// Create dir from input ("D:\a\b\c.ext" or "D:\a\b\", i.e. "D:\a\b")
	static void CreateDir(const string& input);

	/// Get (dir, filebasename, extname) from a full file name
	static vector<string> FileParts(const string& fileName);
	static string GetParentDir(const string& filePath);
	static string GetFileExtension(const string& filePath);
	static string GetFileBaseName(const string& filePath);

	/// Check if a file already existed
	static bool CheckFileExists(const string &fileName);
	/// Check if a dir existed, if not, create it
	static bool VerifyDirectoryExists(const string& directoryName, bool create =
			true);
	/// Delete all files under a dir
	static void ClearDirectory(const string& directoryName);
	/// Get all files under a dir
	static void GetFileList(vector<string>& files, const string& path,
			const string& ext, bool useRootDir = false);
	/// Get all sub-folders
	static void GetSubFolderList(vector<string> & folders, const string & path);

	/// Get current work directory
	static string GetCurrentWorkDirectory();

	/// Add the file separator to filePath if necessary
	static void VerifyTheLastFileSep(string &filePath);

};
/// class FileUtil

}/// namespace RGM

#endif /// RGM_FILEUTILITY_HPP_
