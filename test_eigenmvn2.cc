#include <fstream>
#include <iostream>
#include "eigenmvn.h"


/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 *              1. Full path with csv data samples: with header, coma separated and without index
 *              2. Number of attempts to test
 * @return int 
 */
int main(int argc, char* argv[])
{
        std::string csvFile;
        int trials = 4;        
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << "dataset.csv" << std::endl;
            return 1;
        } else {
            csvFile = argv[1];
            if (argc > 2) {
                trials = atoi(argv[2]);
            }
        }
        std::cout << "File: [" << csvFile << "]\n";
        std::cout << "Trials: [" << trials << "]\n\n\n";

        // create our mvm        
        Eigen::EigenMultivariateNormal<double>  mv_gauss(csvFile);
        std::cout << "Multivariate gaussian built \n";

        // generate some random samples

        // get some rando probs
        Eigen::MatrixXd random_samples = mv_gauss.samples(trials);
        std::cout << "Sample obtained size: " <<  random_samples.rows() << " x " << random_samples.cols() << std::endl;

        Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

        // print sampling outputs 
        // std::cout << random_samples.adjoint().format(CleanFmt) << std::endl;

        // check their probs:
        for (int i = 0; i < trials; ++i){
            Eigen::MatrixXd random_sample = random_samples.col(i);
            std::cout << "P"<< random_sample.adjoint().format(CleanFmt)<< "= ";
            std::cout << mv_gauss.pdf(random_sample) << std::endl;
        }

        return 0;
}
