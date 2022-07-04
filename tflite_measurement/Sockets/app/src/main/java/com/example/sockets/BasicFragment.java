package com.example.sockets;

import android.app.Activity;
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;
import java.io.IOException;
import java.io.InputStream;

/** Basic fragments for the Camera. */
public class BasicFragment extends Fragment {

    /** Tag for the {@link Log}. */
    private static final String TAG = "Check2";
    private TextView textView;
    private ImageClassifier classifier;

    /**
     * Shows a {@link Toast} on the UI thread for the classification results.
     *
     * @param text The message to show
     */
    private void showToast(final String text) {
        final Activity activity = getActivity();
        if (activity != null) {
            activity.runOnUiThread(
                    new Runnable() {
                        @Override
                        public void run() {
                            textView.setText(text);
                        }
                    });
        }
    }

    public static BasicFragment newInstance() {
        return new BasicFragment();
    }

    /** Layout the preview and buttons. */
    @Override
    public View onCreateView(
            LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_basic, container, false);
    }

    /** Connect the buttons to their event handler. */
    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {
        textView = (TextView) view.findViewById(R.id.text);
    }

    /** Load the model and labels. */
    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        try {
            classifier = new ImageClassifier(getActivity());
            classifyImage();
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize an image classifier.");
        }
    }

    @Override
    public void onResume() {
        super.onResume();
    }

    @Override
    public void onPause() {
        super.onPause();
    }

    @Override
    public void onDestroy() {
        classifier.close();
        super.onDestroy();
    }

    /** Classifies a frame from the preview stream. */
    private void classifyImage() {
        if (classifier == null || getActivity() == null) {
            showToast("Uninitialized Classifier or invalid context.");
            return;
        }

        String textToShow = "Good!";
        Log.d(TAG, "1");
//        send2(textToShow);
//        showToast(textToShow);

//        try{
//            AssetManager assetMgr = getActivity().getAssets();
//            InputStream is = assetMgr.open("cat32.png");
//
//            Bitmap bitmap = BitmapFactory.decodeStream(is);
//            String textToShow = classifier.classifyImage(bitmap);
//            bitmap.recycle();
//            Intent i = new Intent(getActivity().getBaseContext(),MainActivity.class);
//            i.putExtra("NAME_KEY", textToShow);
//            showToast(textToShow);
//        } catch(IOException e){
//            e.printStackTrace();
//        }
    }
}