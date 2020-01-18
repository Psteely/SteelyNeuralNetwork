class NeuralNetwork {
    int input, hidden, output;
    float learningRate;


    Matrix weights_IH, weights_HO, bias_H, bias_O;

    NeuralNetwork ( int in_, int hidden_, int out_) {
        input = in_;
        hidden = hidden_;
        output = out_;

        weights_IH = new Matrix(hidden, input);    // weights between in and hidden
        weights_HO = new Matrix(output, hidden);   // weights between hidden and out

        weights_IH.randomise();
        weights_HO.randomise();

        bias_H = new Matrix(hidden, 1);  //weights for the bias for the hidden;
        bias_O = new Matrix(output, 1);  //weights for the bias for the output;

        bias_H.randomise();
        bias_O.randomise();
    }

    float[] predict(float[] inArray_) {
        // get matrix from input array
        Matrix inM = new Matrix(inputArrayToMatrix(inArray_));  // get matrix from input array

        ////// This is our feed forward
        //calculate hidden weights
        Matrix hiddenM = new Matrix(calculateWeights(weights_IH,inM));
        // Take these squished inputs and multiply them by the hidden wights
        Matrix outMatrix = new Matrix(calculateWeights(weights_HO,hiddenM));
        //////

        ///// Handle the output
        float[] ffArray = new float[output];  //  outputs as an array
        ffArray = Matrix.toArray(outMatrix);  //  create Array from outputs
        // Just need to normalise probabilities to total 1 Softmax implementation
        float sum = 0;
        for (int i= 0;i <ffArray.length; i++) {
            ffArray[i] = (float) Math.exp( (float) ffArray[i]);
            sum = sum + ffArray[i];
        }
        for (int i= 0;i <ffArray.length; i++) {
            ffArray[i] = ffArray[i] / sum;
        }
        return ffArray;  // This is our predicted results Array
    }
    void train (float[] inArray_, float[] targets_, float lr_) {
        learningRate = lr_;
        // get matrix from input array
        Matrix inM = new Matrix(inputArrayToMatrix(inArray_));

        // Get the targets as a matrix
        Matrix matrixTargets  = new Matrix(targets_.length, 1);
        matrixTargets.fromArray(targets_)  ;   // targets as a matrix

        /////// This is our feed forward
        //calculate hidden weights
        Matrix hiddenM = new Matrix(calculateWeights(weights_IH,inM));
        // Take these squished inputs and multiply them by the hidden wights
        Matrix outMatrix = new Matrix(calculateWeights(weights_HO,hiddenM));
        ///////

        // This gives us our output errors Matrix.
        Matrix output_errors = new Matrix (Matrix.subMatrix(matrixTargets, outMatrix));

        // Now we have our output errors we need the gradient descent bit.
        // sigmoid derivative
        outMatrix = derivative(outMatrix,output_errors);

        //update bias
        bias_O.addMatrix(outMatrix);

        //calculate deltas
        Matrix weight_HO_deltas = new Matrix(calculateDeltas(outMatrix,hiddenM));

        //Adjust deltas
        weights_HO.addMatrix(weight_HO_deltas);
        // transpose Hidden to output Weights
        Matrix weights_HO_T = new Matrix(Matrix.transpose(weights_HO));
        // get the hidden layers errors.
        Matrix hidden_errors = new Matrix(Matrix.multiply(weights_HO_T, output_errors));
        // hidden gradient
        hiddenM = derivative(hiddenM,hidden_errors);
        // sigmoid derivative
        // adjust bias
        bias_H.addMatrix(hiddenM);

        //  input to hidden
        Matrix weight_IH_deltas = new Matrix(calculateDeltas(hiddenM,inM));

        // adjust weights
        weights_IH.addMatrix(weight_IH_deltas);


    }

    Matrix  calculateWeights (Matrix weights_, Matrix weightsLayer_) {
        // all inputs multiplied by weights
        Matrix Tmp = new Matrix(Matrix.multiply(weights_, weightsLayer_));
        // add in the bias weights
        Tmp.addMatrix(bias_H);
        // activation function on all weights (squish them to 0-1)
        Tmp.sigmoid();
        return Tmp;
    }

    Matrix derivative (Matrix bckM_,Matrix errors_) {
        // Now we have our output errors we need the gradient descent bit.
        // sigmoid derivative
        bckM_.dsigmoid();
        // multiplied by the output errors
        bckM_.multiply(errors_);
        // multiplied by the learning rate
        bckM_.multiply(learningRate);
        //update bias
        return bckM_;

    }

    Matrix calculateDeltas (Matrix out_,Matrix hidden_) {
        //calculate deltas
        Matrix hidden_T = Matrix.transpose(hidden_);
        Matrix weightTmp = new Matrix(Matrix.multiply(out_, hidden_T));
        return weightTmp;
    }

//    void backPropagation (Matrix bckM_, Matrix bckMinus1_,Matrix errors_,Matrix bias_,Matrix weightsToAdjust_) {
//        // Now we have our output errors we need the gradient descent bit.
//        // sigmoid derivative
//        bckM_.dsigmoid();
//        // multiplied by the output errors
//        bckM_.multiply(errors_);
//        // multiplied by the learning rate
//        bckM_.multiply(learningRate);
//        //update bias
//        bias_.addMatrix(bckM_);
//
//        //calculate deltas
//        Matrix hidden_T = Matrix.transpose(bckMinus1_);
//        Matrix weight_HO_deltas = new Matrix(Matrix.multiply(bckM_, hidden_T));
//
//        //Adjust deltas
//        weightsToAdjust_.addMatrix(weightsToAdjust_);
//    }

    Matrix inputArrayToMatrix(float[] inA_) {
        Matrix inMatrix = new Matrix(inA_.length, 1);
        inMatrix.fromArray(inA_);   //   this is now our inputs matrix
        return inMatrix;
    }


    float resultIndex (float[] ra_) {

        float max = -100;
        float actualResult = -100;
        for (int r =0; r<ra_.length; r++) {
            if (ra_[r] > max) {
                max = ra_[r];
                actualResult= r;

            }
        }
        return actualResult;
    }
    float resultProbability (float[] ra_, boolean pcnt_ ) {

        float max = -100;
        float actualResult = -100;
        for (int r =0; r<ra_.length; r++) {
            if (ra_[r] > max) {
                max = ra_[r];


            }
        }
        if (pcnt_){
            return (java.lang.Math.round(max * 100));
        } else return max ;

    }
}