package org.pocl.sample1;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends Activity
{

    // These native functions are defined in jni/vectorAdd.cpp
    public native int initCL();
    public native int vectorAddCL(int N, float[] A, float[] B, float[] C);
    public native int destroyCL();
    public native void setenv(String key, String value);

    TextView text;

    static {
        System.loadLibrary("poclVecAdd");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

        // Forcibly set opencl-stub to use pocl
        setenv("LIBOPENCL_SO_PATH", "/data/data/org.pocl.libs/files/lib/libpocl.so");

        text = new TextView(this);
        text.setText("\nOpenCL vector addition example using pocl\n");

        setContentView(text);

        // Running in separate thread to avoid UI hangs
        Thread td = new Thread() {
            public void run() {
                doVectorAdd();
            }
        };

        td.start();
    }

    void doVectorAdd()
    {
        // Error checkings are not done for simplicity. Check logcat

        printLog("\ncalling opencl init functions... ");
        initCL();

        // Create 2 vectors A & B
        // And yes, this array size is embarrassingly huge for demo!
        float A[] = {1, 2, 3, 4, 5, 6, 7};
        float B[] = {8, 9, 0, 6, 7, 8, 9};
        float C[] = new float[A.length];

        printLog("\n A: ");
        for(int i=0; i<A.length; i++)
            printLog(Float.toString(A[i]) + "    ");

        printLog("\n B: ");
        for(int i=0; i<B.length; i++)
            printLog(Float.toString(B[i]) + "    ");

        printLog("\n\ncalling opencl vector-addition kernel... ");
        vectorAddCL(C.length, A, B, C);

        printLog("\n C: ");
        for(int i=0; i<C.length; i++)
            printLog(Float.toString(C[i]) + "    ");

        boolean correct = true;
        for(int i=0; i<C.length; i++)
        {
            if(C[i] != (A[i] + B[i])) {
                correct = false;
                break;
            }
        }

        if(correct)
            printLog("\n\nresult: passed\n");
        else
            printLog("\n\nresult: failed\n");

        printLog("\ndestroy opencl resources... ");
        destroyCL();
    }

    void printLog(final String str)
    {
        // UI updates should happen only in UI thread
        runOnUiThread(new Runnable() {
             @Override
             public void run() {
                 text.append(str);
             }
        });
    }
}
