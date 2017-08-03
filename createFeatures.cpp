//
// Created by olga on 03.08.17.
//

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

vector<string> FeatchName = {"name", "type", "tata", "ans"};
vector<string> chrName;
vector<int> type;
vector<int> pos;
vector<int> ans;
vector<int> tataDist;
vector<string> seq;

ofstream fout("features.csv");

const int geneLen = 2000;
const int splitPos = 1000;

void printCSV() {
    for (int i = 0; i < (int)FeatchName.size(); ++i) {
        if (i != 0) fout << ",";
        fout << "\"" << FeatchName[i] << "\"";
    }
    fout << "\n";

    for (int i = 0; i < (int)chrName.size(); ++i) {
        fout << "\"" << chrName[i] << "_" << pos[i] << "\",";
        fout << "\"" << type[i] << "\",";
        fout << "\"" << tataDist[i] << "\",";
        fout << "\"" << ans[i] << "\"\n";
    }
}

void readType() {
    ifstream fin("read_type_name");
    string name;
    while (fin >> name) {
        int typ;
        fin >> typ;

        for (int i = 0; i < geneLen; ++i) {
            chrName.push_back(name);

            if (i != splitPos) ans.push_back(0);
            else ans.push_back(1);

            type.push_back(typ);
            pos.push_back(i);
        }
    }
    fin.close();
}

void readGenes() {
    ifstream fin("rice_prom_fgene.1000_1000.2000.fasta");
    string name;

    while (fin >> name) {
        string sq = "";
        string s;
        for (int i = 0; i < 34; ++i) {
            fin >> s;
            sq += s;
        }
        seq.push_back(sq);
    }

    //cerr << seq.size() << " " << seq[0] << "\n";
    fin.close();
}

int findTATAForOne(string& sq, int pos) {
    int res = -1;

    for (int i = max(pos - 40, 0); i < max(pos - 20, 0); ++i) {
        if (sq[i] == 't' && sq[i + 1] == 'a' && sq[i + 2] == 't' && sq[i + 3] == 'a') {
            if (abs(30 - (pos - i)) < abs(30 - res)) {
                res = pos - i;
            }
        }
    }

    return res;

}

void findTATA() {
    for (int i = 0; i < (int)seq.size(); ++i) {
        for (int j = 0; j < geneLen; ++j) {
            tataDist.push_back(findTATAForOne(seq[0], j));
        }
    }
}

int main() {
    readType();
    cerr << "find read type";
    readGenes();
    cerr << "find read";
    findTATA();
    cerr << "find read TATA";
    cerr << chrName.size() << " " << type.size() << " " << pos.size() << " " << ans.size() << " " << tataDist.size() << std::endl;
    printCSV();
}

