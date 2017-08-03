//
// Created by olga on 03.08.17.
//
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <stdio.h>

using namespace std;

int main() {
    ifstream fin("rice_prom_fgene.1000_1000.2000.fasta");
    ofstream fout("read_type_name");
    //type 0 - nothing, 1 - not stress, 2 - stress
    unordered_map <string, int> type;
    vector<string> names;

    string name;

    const int seq_len = 2000;
    const int split_pos = 1000;

    while (fin >> name) {
        string seq;
        for (int i = 0; i < 34; ++i) {
            fin >> seq;
        }

        names.push_back(name);
        type[name] = 1;
    }

    ifstream fin2("rice_go.stress.csv");

    string info;

    while (getline(fin2, info)) {
        if (info[1] != 'c') {
            continue;
        }
        string res_n = ">";
        for (int i = 1; info[i] != '"'; ++i) {
            res_n += info[i];
        }

        type[res_n] = 2;

        cerr << res_n << "\n";
    }

    for (int i = 0; i < (int)names.size(); ++i) {
        fout << names[i] << "\n";
        fout << type[names[i]] << "\n";
    }

}