package com.example.sockets;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

public class MainActivity extends AppCompatActivity {
    EditText e1;
    TextView textView;
    private ImageClassifier classifier;
    String message = "Yes";

    private static final String TAG = "Check";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        e1 = (EditText)findViewById(R.id.editText);
        textView = (TextView)findViewById(R.id.textView);

        Thread myThread = new Thread(new MyServerThread());
        myThread.start();
    }

    class MyServerThread implements Runnable {
        Socket s;
        ServerSocket ss;
        InputStreamReader isr;
        BufferedReader bufferedReader;
        Handler h = new Handler();
        @Override
        public void run() {
            try {
                ss = new ServerSocket(7801);
                while(true){
                    s = ss.accept();
                    isr = new InputStreamReader(s.getInputStream());
                    bufferedReader = new BufferedReader(isr);
                    h.post(new Runnable() {
                        @Override
                        public void run() {
                            classifyImageHandler();
                        }
                    });
                }
            } catch (IOException e){
                e.printStackTrace();
            }
        }
    }

    //                    message = bufferedReader.readLine();
//                    Log.d(TAG, "5");

    /** Classifies a frame from the preview stream. **/
    private void classifyImageHandler() {
        try {
            try {
//                Log.d(TAG, "5");
                classifier = new ImageClassifier(MainActivity.this);
            } catch (IOException e) {
                e.printStackTrace();
            }

            float total_time = 0;
            int test_num = 128; //74;
            int warm_up = 64; //10;

            for (int i = 0; i<test_num; i++) {
                AssetManager assetMgr = MainActivity.this.getAssets();
                InputStream is = assetMgr.open("cat224.png");
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                message = classifier.classifyImage(bitmap);
                bitmap.recycle();
                if (i >= warm_up) {
                    total_time += Float.parseFloat(message);
                }
            }
//            float fps = (test_num - warm_up) * (1000000000 / total_time);
            float duration = (total_time / 1000000) * ((float)1 / (test_num - warm_up));
            message = Float.toString(duration);  //duration
//            textView.setText(message);
            textView.append(message + "\n");
            send2(message);
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public void send(View v){
        try {
            try {
                classifier = new ImageClassifier(MainActivity.this);
            } catch (IOException e) {
                e.printStackTrace();
            }

            float total_time = 0;
            int test_num = 128; // 74;
            int warm_up = 64; //10;

            for (int i = 0; i<test_num; i++){
                AssetManager assetMgr = MainActivity.this.getAssets();
                InputStream is = assetMgr.open("cat224.png");
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                message = classifier.classifyImage(bitmap);
                bitmap.recycle();
                if (i >= warm_up){
                    total_time += Float.parseFloat(message);
                }
            }
//            float fps = (test_num - warm_up) * (1000000000 / total_time);
            float duration = (total_time / 1000000) * ((float)1 / (test_num - warm_up));
            message = Float.toString(duration);
//            textView.setText(message);
            textView.append(message + "\n");
        } catch (IOException e){
            e.printStackTrace();
        }
        MessageSender messageSender = new MessageSender();
        messageSender.execute(message);
    }
    public void send2(String s){
        MessageSender messageSender = new MessageSender();
        messageSender.execute(s);
    }
}