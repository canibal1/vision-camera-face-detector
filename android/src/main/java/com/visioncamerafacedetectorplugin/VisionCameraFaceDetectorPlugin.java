package com.visioncamerafacedetectorplugin;

import android.graphics.Bitmap;
import android.graphics.PointF;
import android.graphics.Rect;
import android.media.Image;
import android.util.Base64;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.android.gms.tasks.Task;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;
import com.mrousavy.camera.frameprocessor.Frame;
import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin;

import com.visioncamerafacedetectorplugin.models.FaceDetectionStatus;
import com.visioncamerafacedetectorplugin.models.FaceDetectorException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class VisionCameraFaceDetectorPlugin extends FrameProcessorPlugin {
  private static final String START = "startFaceDetector";
  private static final String CLOSE = "closeFaceDetector";
  Map<String, Object> resultMap = new HashMap<>();
  private long lastDetectionTime = 0;
  FaceDetector faceDetector;
  private final Map<String, com.google.mlkit.vision.face.FaceDetector> instances = new HashMap<>();

  FaceDetectorOptions faceDetectorOptions =
    new FaceDetectorOptions.Builder()
      .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
      .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
      .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
      .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
      .enableTracking()
      .setMinFaceSize(0.9f)
      .build();

  private void setErrorResult(String error) {
    resultMap.put("status", FaceDetectionStatus.ERROR.name().toLowerCase());
    resultMap.put("error", error);
    resultMap.put("faces", null);
  }

  VisionCameraFaceDetectorPlugin(@Nullable Map<String, Object> options) {
    super(options);
  }


  @Override
  public Object callback(@NonNull Frame frame, @Nullable Map<String, Object> params) {
    String method = params != null && params.get("status") != null ? params.get("status").toString() : "";
    switch (method) {
      case START:
        return handleDetection(frame, params);
      case CLOSE:
        return closeDetector(params);
      default:
        return resultMap;
    }


  }

  private Map<String, Object> handleDetection(@NonNull Frame frame, @Nullable Map<String, Object> params) {
    long currentTime = System.currentTimeMillis();

    if (currentTime - lastDetectionTime < 1500) {
      return resultMap;
    }
    lastDetectionTime = currentTime;
    try {
      String id = params != null && params.get("id") != null ? params.get("id").toString() : "";
      Image mediaImage = frame.getImage();
      if (mediaImage == null) {
        throw new FaceDetectorException(103, "Cannot get image from frame");
      }
      faceDetector = instances.get(id);

      if (faceDetector == null) {
        faceDetector = FaceDetection.getClient(faceDetectorOptions);
        instances.put(id, faceDetector);
      }
      Log.e("faceDetector.hashCode()", String.valueOf(faceDetector.hashCode()));
      InputImage image = InputImage.fromMediaImage(mediaImage, Utils.convertRotationDegreeFromString(frame.getOrientation()));
      faceDetector.process(image).addOnSuccessListener(
        visionFaces -> {
          List<Map<String, Object>> faces = new ArrayList<>(visionFaces.size());

          if (visionFaces.size() == 1) {
            for (Face face : visionFaces) {
              Map<String, Object> faceData = new HashMap<>();


              faceData.put("headEulerAngleY", face.getHeadEulerAngleY());

              if (face.getSmilingProbability() != null) {
                faceData.put("smilingProbability", face.getSmilingProbability());
              }

              if (face.getLeftEyeOpenProbability() != null) {
                faceData.put("leftEyeOpenProbability", face.getLeftEyeOpenProbability());
              }

              if (face.getRightEyeOpenProbability() != null) {
                faceData.put("rightEyeOpenProbability", face.getRightEyeOpenProbability());
              }

              if (face.getTrackingId() != null) {
                faceData.put("trackingId", face.getTrackingId());
              }
              faces.add(faceData);
            }
            resultMap.put("faces", faces.toString());
            resultMap.put("status", FaceDetectionStatus.SUCCESS.name().toLowerCase());
            resultMap.put("error", null);
          } else {
            setErrorResult(new FaceDetectorException(104, "Faces not found").toString());

          }
        }

      ).addOnFailureListener(
        (Exception e) -> {
          Log.e("FaceDetectionError", e.getMessage());
          setErrorResult(e.toString());
        }
      );

      return resultMap;

    } catch (FaceDetectorException e) {
      Log.e("FaceDetectionError", e.getMessage());
      setErrorResult(e.toString());
      return resultMap;
    } catch (Exception e) {
      Log.e("FaceDetectionError", e.toString());
      setErrorResult(new FaceDetectorException(101, "System Error").toString());
      return resultMap;
    }
  }

  private Map<String, Object> closeDetector(@Nullable Map<String, Object> params) {
    String id = params != null && params.get("id") != null ? params.get("id").toString() : "";
    resultMap = new HashMap<>();
    com.google.mlkit.vision.face.FaceDetector detector = instances.get(id);
    if (detector == null) return resultMap;
    detector.close();
    instances.remove(id);
    return resultMap;
  }


}
