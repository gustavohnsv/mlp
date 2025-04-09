#ifndef DATASETS_H
#define DATASETS_H

#include <cmath>
#include <vector>
#include <stdexcept>

#include "./Parameters/params.h"

class Datasets {
    private:
        std::vector<Params> params;
        int size;
        int size_params;
        std::vector<double> means;
        std::vector<double> std_devs;
        bool isStandardized;
    public:
        Datasets(int size, int size_params);
        Datasets();
        ~Datasets();
        void insertParams(std::vector<double> x, int y, int size_p, size_t index);
        Params getUniqueParams(size_t index);
        int getSize();
        size_t getSizeParams();
        void standardize();
        bool isStandardizedParams() const;
        void standardizeParams(Params &params);
};

#endif // DATASETS_H