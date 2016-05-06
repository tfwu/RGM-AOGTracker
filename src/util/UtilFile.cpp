#include <boost/foreach.hpp>

#include "util/UtilFile.hpp"

namespace RGM {

bool FileUtil::exists(const string & input) {
	return fs::exists(fs::path(input));
}

void FileUtil::CreateDir(const string& input) {
	fs::path p(input.c_str());

    if(!fs::is_directory(p)) {

		fs::path pp = p.parent_path();
		fs::create_directories(pp);
	}
}

vector<string> FileUtil::FileParts(const string& fileName) {
	fs::path p(fileName.c_str());

	vector < string > vstr(3);

	vstr[0] = p.parent_path().string();
	vstr[1] = p.stem().string();
	vstr[2] = p.extension().string();

	return vstr;
}

string FileUtil::GetParentDir(const string& filePath) {
	string p = fs::path(filePath.c_str()).parent_path().string();
	VerifyTheLastFileSep(p);
	return p;
}

string FileUtil::GetFileExtension(const string& filePath) {

	return fs::path(filePath.c_str()).extension().string();
}

string FileUtil::GetFileBaseName(const string& filePath) {

	return fs::path(filePath.c_str()).stem().string();
}

bool FileUtil::CheckFileExists(const string& fileName) {

	return fs::exists(fs::path(fileName.c_str()));
}

bool FileUtil::VerifyDirectoryExists(const string& directoryName, bool create) {

	fs::path p(directoryName.c_str());

	bool exist = fs::exists(p);

    if(!exist && create) {
		return fs::create_directories(p);
	}

	return exist;
}

void FileUtil::ClearDirectory(const string& directoryName) {

	fs::path p(directoryName.c_str());

	fs::remove_all(p);
	fs::create_directory(p);
}

void FileUtil::GetFileList(vector<string>& files, const string& path,
                           const string& ext, bool useRootDir) {

    // ext include ., e.g., ".jpg"
    if (!exists(path)) return;

    fs::path targetDir(path.c_str());

	fs::directory_iterator it(targetDir), eod;

	bool getAllFiles = (ext.compare("*") == 0);

    BOOST_FOREACH(fs::path const & p, make_pair(it, eod)) {

        if(fs::is_regular_file(p)) {
            if(getAllFiles) {
                useRootDir ? files.push_back(p.string()) :
				files.push_back(p.filename().string());
            } else {
                if(ext.compare(p.extension().string()) == 0) {
                    useRootDir ? files.push_back(p.string()) :
                    files.push_back(p.filename().string());
                }
            }
		}
	}
}

void FileUtil::GetSubFolderList(vector<string> & folders, const string & path) {

    folders.clear();

    fs::path targetDir(path.c_str());

    fs::directory_iterator it(targetDir), eod;

    BOOST_FOREACH(fs::path const & p, std::make_pair(it, eod)) {

        if(fs::is_directory(p)) {
            //std::string tmp = p.filename().string();
            folders.push_back(p.filename().string());
        }
    }

    std::sort(folders.begin(), folders.end());
}

string FileUtil::GetCurrentWorkDirectory() {
//#if  (defined(WIN32)  || defined(_WIN32) || defined(WIN64) || defined(_WIN64))
//	char buffer[PATH_MAX];
//	GetModuleFileName( NULL, buffer, PATH_MAX );
//	string::size_type pos = string( buffer ).find_last_of( "\\/" );
//	return string( buffer ).substr( 0, pos);
//#else
//	char szTmp[32];
//	sprintf(szTmp, "/proc/%d/exe", getpid());
//	int bytes = min(readlink(szTmp, pBuf, len), len - 1);
//	if(bytes >= 0)
//		pBuf[bytes] = '\0';
//	return bytes;
//#endif

    fs::path p = fs::initial_path();
    return p.string();
}

void FileUtil::VerifyTheLastFileSep(string &filePath) {
    if ( filePath.empty() ) return;

    if(filePath.find_last_of("/\\") != (filePath.length() - 1)) {
        filePath = filePath + FILESEP;
    }
}

} // namespace RGM

