#ifndef PARAMS_H
#define PARAMS_H

#include <vector>

class Params {
    private:
        std::vector<double> x;
        int y;
        int size;
    public:
        Params(int size);
        Params();
        void insertX(double value, size_t index);
        void insertY(int value);
        double getUniqueX(size_t index);
        int getY();
        size_t getSize();
};

#endif // PARAMS_H