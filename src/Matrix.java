
public class Matrix { //
    int rows ;
    int cols ;
    float elements[][];


    Matrix (int r_, int c_) {  //  standard constructor
        rows = r_;
        cols = c_;
        elements = new float[rows][cols];
        for (int r= 0; r <rows; r++) {
            for  (int c= 0; c <cols; c++) {
                elements[r][c] =  0;
            }
        }
    }

    Matrix (Matrix m) {  // constructor for static multiplication method
        rows = m.rows;
        cols = m.cols;
        elements = m.elements;
    }


    //////////////////////////////////////

    void fromArray( float[] in_ ) {
        for (int r= 0; r <in_.length; r++) {
            for  (int c= 0; c <1; c++) {
                elements[r][c] = in_[r];
            }
        }
    }

    static float[] toArray( Matrix in_ ) {
        float[] result = new float[in_.rows];
        if (in_.cols > 1) {

            System.out.println("Can only convert single column matrix to array");
            return result;
        }
        for (int r= 0; r <in_.rows; r++) {
            for  (int c= 0; c <1; c++) {
                result[r] = in_.elements[r][c];
            }
        }
        return result;
    }
    // Randomise
    void randomise() {
        for (int r= 0; r <rows; r++) {
            for  (int c= 0; c <cols; c++) {
                elements[r][c] =  (float)((Math.random()*2.0)-1);
            }
        }
    }

    ///   Print matrix for debugging

    void logM() {

        //processing.core.PApplet.stroke(0);
        System.out.println(rows + "X" + cols);
        System.out.println("------------------");
        for (int r=0; r<rows; r++) { // for every column
            for (int c=0; c<cols; c++) { // for every row
                System.out.print(elements[r][c] + " ");
            }
            System.out.println();
        }
    }

    // matrix maths methods
    //////////////////////////////////

    void addMatrix ( float n) {
        for (int r= 0; r <rows; r++) {
            for  (int c= 0; c <cols; c++) {
                elements[r][c] +=  n;
            }
        }
    }
    //element functions.
    void addMatrix ( Matrix n) {
        for (int r= 0; r <rows; r++) {
            for  (int c= 0; c <cols; c++) {
                elements[r][c] +=  n.elements[r][c];
            }
        }
    }
    static Matrix subMatrix ( Matrix a, Matrix b) {
        Matrix result = new Matrix(a.rows,a.cols);
        for (int r= 0; r <a.rows; r++) {
            for  (int c= 0; c <a.cols; c++) {
                result.elements[r][c] =  a.elements[r][c] - b.elements[r][c];
            }
        }
        return result;
    }




    // scalar element wise multiply
    void multiply ( float n) {
        //matrix mt = new matrix(rows,cols);
        for (int r= 0; r <rows; r++) {
            for  (int c= 0; c <cols; c++) {
                elements[r][c] *=  n;
            }
        }
    }
    // element multiplication
    void multiply ( Matrix n) {
        for (int r= 0; r <rows; r++) {
            for  (int c= 0; c <cols; c++) {
                elements[r][c] *=  n.elements[r][c];
            }
        }
    }
    //Dot Prod function. Static multiplication
    static Matrix  multiply ( Matrix a, Matrix b) {
        if (a.cols != b.rows) {
            System.out.println("Cols of A not = Rows of B");
            return null;
        }
        Matrix result = new Matrix(a.rows, b.cols);  // new matrix
        for (int i = 0; i < result.rows; i ++) {
            for (int j = 0; j < result.cols; j ++) {
                float sum = 0;
                for (int k = 0; k<a.cols; k++) {
                    sum += a.elements[i][k] * b.elements[k][j];
                }
                result.elements[i][j] =sum;
            }
        }
        return result;
    }
    static Matrix transpose (Matrix a) {
        //int newRows = cols;
        //int newCols = rows;
        Matrix result = new Matrix(a.cols, a.rows);  // new matrix
        for (int r= 0; r <a.rows; r++) {
            for  (int c= 0; c <a.cols; c++) {
                result.elements[c][r] =  a.elements[r][c];
            }
        }
        return result;
    }

    void sigmoid() {
        for (int r= 0; r <rows; r++) {
            for  (int c= 0; c <cols; c++) {
                elements[r][c] = (float) (1 / (1 + Math.exp(-elements[r][c])));
            }
        }
    }
    void dsigmoid() {
        for (int r= 0; r <rows; r++) {
            for  (int c= 0; c <cols; c++) {
                elements[r][c] *= (1-elements[r][c]);
            }
        }
    }
}