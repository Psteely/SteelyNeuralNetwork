public class MainClass {
    public static void main(String[] args) {

    }

    NeuralNetwork nn;



public void setup() {

    nn = new NeuralNetwork(2,2,2);
    System.out.println("XX");
}
}

//   Use this stuff if we need to test the changes we make to the jar file.


//
//
//import processing.core.PApplet;
//import processing.data.Table;
//import processing.data.TableRow;
//
//import java.util.ArrayList;
//
//public class MainClass extends PApplet {
//    public static PApplet processing;
//    public static void main(String[] args) {
//        PApplet.main("MainClass", args);
//    }
//
////    public void setup() {
////        processing = this;
////    }
//
//    //<>//
//
//    public ArrayList<float[]> errors = new ArrayList<float[]>() ;
//
//    // MNIST csv are here https://pjreddie.com/projects/mnist-in-csv/
//    NeuralNetwork nn;
//
//    int in = 784;
//    int middle = 256;
//    int out = 10;
//    int pixelSize = 20;
//    float[][] training;
//    float[][] TrainingLabel;
//    float[][] testing;
//    float[][] TestingLabel;
//    int errorNum = 0;
//    float lastKeyPressed;
//    float[] predictionImage = new float[784];
//
//
//    Table csvReader;
//    TableRow csvRow;
//
//
//
//
//    public void setup() {
//        processing = this;
//        background(0);
//        surface.setSize(28 * pixelSize, 28 * pixelSize);  // only for showing the number
//        //line(28,0,28,28);
//        nn = new NeuralNetwork(in, middle, out);
//        train(1);
////        train(2);
////        train(3);
//
//      test();
//
//    }
//
//
//    public void draw() {
//
//    }
//
//    void showError () {
//
//        background(0);
//        if (errors != null) {
//            float[] pic;
//            pic = errors.get(errorNum);
//            for (int i = 0; i < 784; i++) {
//                float val = pic[i+1];
//                fill(val*255);
//                rect(pixelSize * (i % 28), pixelSize * (i / 28), pixelSize, pixelSize);
//
//            }
//            println(pic[0]);
//
//        }
//
//    }
//
//    void showPrediction () {
//
//        background(0);
//
//        float[] pic;
//        pic = predictionImage;
//        for (int i = 0; i < 784; i++) {
//            float val = predictionImage[i];
//
//            stroke(val);
//            fill(val);
//
//            rect(pixelSize * (i % 28), pixelSize * (i / 28), pixelSize, pixelSize);
//
//        }
//        float[] result = new float[10];
//        for (int i =0; i < predictionImage.length; i ++) {
//            predictionImage[i] = predictionImage[i] /255;
//
//            //println(predictionImage.length);
//            result = nn.predict(predictionImage);      // predict that network
//
//
//
//
//        }
//
//        println("Prediction = " + nn.resultIndex(result) +" percentage = " + nn.resultProbability(result,true)); // print out result index
//
//    }
//
//
////    public void keyPressed() {
//
////    }
//
//
//
//    public void mouseDragged() {
//
//
//        fill(255);
//        stroke(255);
//        strokeWeight(20);
//        line(pmouseX,pmouseY,mouseX,mouseY);
//
//
//    }
//
//
//    public void keyPressed() {
//
//        if (lastKeyPressed != 32) {
//            if (keyCode == 32) {     //  only if space bar
//                lastKeyPressed = keyCode;
//
//                //loadPixels();
//                float totalforBox = 0;
//                int predictionCount = 0;
//                for (int c = 0; c < 28; c++) {        // loop along each column
//                    for (int r = 0; r < 28; r++) {     // loop along each row
//                        totalforBox = 0;               //clear the box
//                        for (int pc = 0; pc < pixelSize; pc++) {     // loop along each pixel in the box
//                            for (int pr = 0; pr < pixelSize; pr++) { // loop along each row in the box
//                                int co = get((r * pixelSize + pr), (c * pixelSize + pc));
//                                float red = MainClass.processing.red(co);
//                                float blue = MainClass.processing.blue(co);
//                                float green = MainClass.processing.green(co);
//                                int grey = (int) (red + blue + green) / 3;
//                                totalforBox = totalforBox + grey; //add upp the pixels in the box
//                            }
//                        }
//                        predictionImage[predictionCount] = totalforBox / (pixelSize * pixelSize);
//                        if (predictionImage[predictionCount] > 255) {
//                            predictionImage[predictionCount] = 255;
//                        }
//                        predictionCount++;
//                    }
//
//
//                }
//
//
//                showPrediction();
//            }
//        }
//        if (keyCode == 67) {
//            background(0);
//        }
//
////            if (keyCode == 39) {
////                errorNum  ++;
////
////            } else errorNum --;
////            if (errorNum <0) {
////                errorNum=0;
////
////            }
//
//
//        lastKeyPressed = keyCode;
//    }
//
//
//
//
//    //import java.util.Arrays;
//    void test () {
//
//
//        csvReader = null;
//
//        println("Testing table load starts");
//        csvReader = loadTable("../../bigdata/mnist_test.csv");      // read in the file
//        println("Testing table load ends");
//
//
//        //noStroke();
//        //currentRow = (int) random(csvReader.getRowCount()-1);
//        testing = new float[csvReader.getRowCount()][784];   // set up training data array
//        TestingLabel = new float[csvReader.getRowCount()][out];      // set up label array
//        for (int r = 0; r< csvReader.getRowCount(); r++) {    // loop around each row for as manay as we have BIG OUTER LOOP
//            csvRow = csvReader.getRow(r);                       // get the current row
//            for (int l =0; l<out; l++) {                        // loop aroud the label array 10 in this case
//                if (l==csvRow.getInt(0)) {                        // if array is the right one ...
//                    TestingLabel[r][l] = 1;                                // ... stick a 1 in it
//                } else TestingLabel[r][l] = 0 ;                          // else a zero
//            }
//
//            for (int c = 0; c < csvReader.getColumnCount()-1; c++) { // now run around the colums 784 of them.
//                float val = csvRow.getInt(c+1);                        // grab the vale for each column or pixel
//                testing[r][c] = val/255;                              // Add it to the training array /255 to normalise between 0 and 1
//            }
//        }
//        // set up the nn
//        float numRight=0;
//        float percent = 0;
//        float expected = 100;
//        float actual = 100;
//        float actualResult = -1;
//        float[] result = new float[10];
//        float[] in = new float[784];                               // array for inputs
//        float[] lab = new float[10];                               // array for labels
//        println("Testing begins");
//        for (int i = 0; i <csvReader.getRowCount(); i++) {         // loop around each row
//            //for (int i = 0; i <1; i++) {                               // loop around one row
//            for (int j = 0; j <784; j++) {                           // grab each training array in a loop (must be a better way to do this
//                in[j] = testing[i][j];                                  // set the input array
//            }
//            for (int j = 0; j <10; j++) {                            // grab each label array in a loop (must be a better way to do this
//                lab[j] = TestingLabel[i][j];                                  // set the label array
//            }
//            result = nn.predict(in);      // predict that network
//            //Whats the expected result
//            for (int r =0; r<lab.length; r++) {                     // whats the expected number
//                if (lab[r] == 1.0) {
//                    expected = r;
//                }
//            }
//            //Whats the actual result
//            float max =-100;
//
//            for (int r =0; r<result.length; r++) {
//                if (result[r] > max) {
//                    max = result[r];
//                    actualResult= r;
//                }
//                //println(result);
//            }
//            // whats the percentage
//
//
//            if (expected == actualResult) {
//                numRight ++;
//            } else {
//                float[] err = new float[785];
//                err[0] = actualResult;
//                for (int e=0; e<784; e++) {
//                    err[e+1] = in[e];
//                }
//                errors.add(err);
//
//            }
//
//        }
//        percent = (numRight / csvReader.getRowCount()) * 100;
//        println("Testing complete");
//        println("Percentage right = " + percent);
//        println("Number of errors = " + errors.size());
//    }
//
//    void train (int num) {
//
//        csvReader = null;
//        if (num == 1) {
//            println("Training table A load starts");
//            csvReader = loadTable("../../bigdata/mnist_train_20000b.csv");      // read in the file
//        } else if (num == 2) {
//            println("Training table B load starts");
//            csvReader = loadTable("../../bigdata/mnist_train_20000b.csv");      // read in the file
//        } else if (num == 3) {
//            println("Training table C load starts");
//            csvReader = loadTable("../../bigdata/mnist_train_20000c.csv");      // read in the file
//        }
//
//        println("Training table load ends");
//        //noStroke();
//        //currentRow = (int) random(csvReader.getRowCount()-1);
//        training = new float[csvReader.getRowCount()][784];   // set up training data array
//        TrainingLabel = new float[csvReader.getRowCount()][out];      // set up label array
//        for (int r = 0; r< csvReader.getRowCount(); r++) {    // loop around each row for as manay as we have BIG OUTER LOOP
//            csvRow = csvReader.getRow(r);                       // get the current row
//            for (int l =0; l<out; l++) {                        // loop aroud the label array 10 in this case
//                if (l==csvRow.getInt(0)) {                        // if array is the right one ...
//                    TrainingLabel[r][l] = 1;                                // ... stick a 1 in it
//                } else TrainingLabel[r][l] = 0 ;                          // else a zero
//            }
//
//            for (int c = 0; c < csvReader.getColumnCount()-1; c++) { // now run around the colums 784 of them.
//                float val = csvRow.getInt(c+1);                        // grab the vale for each column or pixel
//                training[r][c] = val/255;                              // Add it to the training array /255 to normalise between 0 and 1
//            }
//        }
//        // set up the nn
//
//        float[] in = new float[784];                               // array for inputs
//        float[] lab = new float[10];                               // array for labels
//        println("Training begins");
//        for (int i = 0; i <csvReader.getRowCount(); i++) {         // loop around one row
//            for (int j = 0; j <784; j++) {                           // grab each training array in a loop (must be a better way to do this
//                in[j] = training[i][j];                                  // set the input array
//            }
//            for (int j = 0; j <10; j++) {                            // grab each label array in a loop (must be a better way to do this
//                lab[j] = TrainingLabel[i][j];                                  // set the label array
//            }
//            nn.train(in, lab, 0.1f );                                 // train that network
//        }
//        println("Training complete");
//    }
//
//}