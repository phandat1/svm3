#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <graphics.h>

using namespace std;

struct DataPoint {
    vector<double> features;
    int label;
};

double dotProduct(const vector<double>& v1, const vector<double>& v2) {
    double result = 0.0;
    for (int i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

void initializeWeights(vector<double>& w, int featureSize) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-0.1, 0.1);
    w.clear();
    for (int i = 0; i < featureSize; ++i) {
        w.push_back(dis(gen));
    }
}

vector<double> trainSVM(vector<DataPoint>& data, double learningRate = 0.01, int epochs = 10000, double lambda = 0.01) {
    if (data.empty()) return {};

    int featureSize = data[0].features.size();
    vector<double> w(featureSize, 0.0);
    double b = 0.0;

    initializeWeights(w, featureSize); // Initialize weights randomly

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (auto& point : data) {
            if (point.label * (dotProduct(w, point.features) + b) < 1) {
                for (int i = 0; i < featureSize; ++i) {
                    w[i] -= learningRate * (lambda * w[i] - point.label * point.features[i]); // Adding L2 regularization term
                }
                b -= learningRate * (-point.label);
            }
        }
        learningRate *= 0.99; // Gradually decrease the learning rate
    }
    w.push_back(b); // Append bias to the weight vector for simplicity
    return w; // Returning w and b as part of the same vector
}

void drawGraph(const vector<DataPoint>& data, const vector<double>& w) {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, NULL); // Initialize graphics

    // Draw data points
    for (const auto& point : data) {
        int x = 200 + 30 * point.features[0]; // Scale coordinate x
        int y = 200 - 30 * point.features[1]; // Scale coordinate y
        if (point.label == 1)
            setcolor(2); // Green for label 1
        else
            setcolor(4); // Red for label -1
        circle(x, y, 5); // Draw circle at (x, y)
        floodfill(x, y, getcolor());
    }

    // Draw the hyperplane
    if (w.size() >= 3) {
        setcolor(15); // White color
        for (int x = 0; x < getmaxx(); x++) {
            double x1 = (x - 200) / 30.0;
            double y1 = (-w[2] - w[0] * x1) / w[1];
            int vy = 200 - 30 * y1;
            if (vy >= 0 && vy <= getmaxy()) {
                putpixel(x, vy, WHITE);
            }
        }
    }

    getch(); // Wait for user input
    closegraph(); // Close graphics
}

int main() {
    vector<DataPoint> data = {
        {{3, 4}, 1},{{2.5,5},1},{{1.5,3.5},1},{{2.5,6},1},{{3.5,5},1},{{4,6},1}, {{4, 5}, 1}, {{5, 3}, -1},
	{{3, 1}, -1},{{3.5, 1}, -1}, {{3.5, 2.5}, -1}, {{5.5, 2}, -1}, {{4.5,2.5},-1},{{4, 2.5}, -1},{{5, 2.3}, -1}
    };

    vector<double> svmParameters = trainSVM(data);

    cout << "Model parameters: ";
    for (int i = 0; i < svmParameters.size() - 1; ++i) {
        cout << "w" << i << " = " << svmParameters[i] << ", ";
    }
    cout << "b = " << svmParameters.back() << endl;
    cout << "Final SVM Hyperplane: y = ";
    cout << svmParameters[0] << "*x" << 1 ;
    for (int i = 1; i < svmParameters.size() - 1; ++i){
    	if(svmParameters[i]>0)
        cout << " + " <<svmParameters[i]<< "*x" << i+1 ;
        else cout<<" - "<<abs(svmParameters[i])<<"*x"<<i+1;
        
    }
    cout << svmParameters.back() << endl;

    drawGraph(data, svmParameters);

    return 0;
}
