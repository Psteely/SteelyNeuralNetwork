class NeuralNetwork {
    int input, hidden, output;


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
        // generate hidden outputs
        Matrix inMatrix = new Matrix(inArray_.length, 1);
        inMatrix.fromArray(inArray_);   //   this is now our inputs matrix
        // all inputs multiplied by weights
        Matrix hidden = new Matrix(Matrix.multiply(weights_IH, inMatrix));
        // add in the bias weights
        hidden.addMatrix(bias_H);

        // activation function on all weights (squish them to 0-1)
        hidden.sigmoid();
        // Take these squished inputs and multiply them by the hidden wights
        Matrix outMatrix = new Matrix(Matrix.multiply(weights_HO, hidden));
        // add in the output bias
        outMatrix.addMatrix(bias_O);
        //squish it again. This is now our guess. Between 0-1
        outMatrix.sigmoid();
        // return as an array
        float[] ffArray = new float[output];
        ffArray = Matrix.toArray(outMatrix);
        // Just need to normalise probabilities to total 1 Softmax implementation
        float sum = 0;
        for (int i= 0;i <ffArray.length; i++) {
            ffArray[i] = (float) Math.exp( (float) ffArray[i]);
            sum = sum + ffArray[i];
        }
        for (int i= 0;i <ffArray.length; i++) {
            ffArray[i] = ffArray[i] / sum;

        }
        return ffArray;
    }
    void train (float[] inArray_, float[] targets_, float lr_) {
        // ************  From here ***********************************
        // *********** is really just the feed forward bit ***********
        // generate hidden outputs
        Matrix inputs = new Matrix(inArray_.length, 1);
        inputs.fromArray(inArray_);   //   this is now our inpus matrix
        // all inputs multiplied by weights
        Matrix hidden = new Matrix(Matrix.multiply(weights_IH, inputs));
        // Add in the bias weight
        hidden.addMatrix(bias_H);

        // activation function squish to 0-1
        hidden.sigmoid();
        // Generating output outputs.
        Matrix outMatrix = new Matrix(Matrix.multiply(weights_HO, hidden));
        // Take the squished hidden and multiply by the hidden weights.
        outMatrix.addMatrix(bias_O);
        //Squish it
        outMatrix.sigmoid();

        // ************  to here  **************************************
        // Now we need to do the back propogation
        // Get the targets as a matrix
        Matrix matrixTargets  = new Matrix(targets_.length, 1);
        matrixTargets.fromArray(targets_)  ;   // targets as a matrix
        // This gives us our output errors matrix.
        Matrix output_errors = new Matrix (Matrix.subMatrix(matrixTargets, outMatrix));

        // Now we have our output errors we need the gradient descent bit.
        // sigmoid derivative
        outMatrix.dsigmoid();
        // multiplied by the output errors
        outMatrix.multiply(output_errors);
        // multiplied by the learning rate
        outMatrix.multiply(lr_);
        //update bias
        bias_O.addMatrix(outMatrix);

        //calculate deltas
        Matrix hidden_T = Matrix.transpose(hidden);
        Matrix weight_HO_deltas = new Matrix(Matrix.multiply(outMatrix, hidden_T));

        //Adjust deltas
        weights_HO.addMatrix(weight_HO_deltas);

        // transpose Hidden to output Weights
        Matrix weights_HO_T = new Matrix(Matrix.transpose(weights_HO));
        // get the hidden layers errors.
        Matrix hidden_errors = new Matrix(Matrix.multiply(weights_HO_T, output_errors));
        // hidden gradient
        // sigmoid derivative
        hidden.dsigmoid();
        hidden.multiply(hidden_errors);
        hidden.multiply(lr_);
        // adjust bias
        bias_H.addMatrix(hidden);

        //  input to hidden
        Matrix inputs_T = new Matrix(Matrix.transpose(inputs));
        Matrix weight_IH_deltas = new Matrix(Matrix.multiply(hidden,inputs_T));

        // adjust
        weights_IH.addMatrix(weight_IH_deltas);


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