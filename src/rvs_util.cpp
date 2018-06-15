// Copyright [year] <Copyright Owner> ... goes here
#include <vector>
#include <string>

#include "rvs_util.h"

using std::vector;
using std::string;

/**
 * splits a string based on a given delimiter
 * @param str_val input string
 * @param delimiter tokens' delimiter
 * @return vector containing all tokens
 */
vector<string> str_split(const string& str_val, const string& delimiter) {
    vector<string> str_tokens;
    unsigned int prev_pos = 0, cur_pos = 0;
    do {
        cur_pos = str_val.find(delimiter, prev_pos);
        if (cur_pos == string::npos)
            cur_pos = str_val.length();
        string token = str_val.substr(prev_pos, cur_pos - prev_pos);
        if (!token.empty())
            str_tokens.push_back(token);
        prev_pos = cur_pos + delimiter.length();
    } while (cur_pos < str_val.length() && prev_pos < str_val.length());
    return str_tokens;
}
