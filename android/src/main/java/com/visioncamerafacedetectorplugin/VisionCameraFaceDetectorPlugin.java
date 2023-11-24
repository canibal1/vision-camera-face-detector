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
import com.visioncamerafacedetectorplugin.models.FaceDirection;

import java.io.ByteArrayOutputStream;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;


public class VisionCameraFaceDetectorPlugin extends FrameProcessorPlugin {
  private static final String START = "startFaceDetector";
  private static final String CLOSE = "closeFaceDetector";
  Map<String, Object> resultMap = new HashMap<>();
  private long lastDetectionTime = 0;
  private final Map<String, com.google.mlkit.vision.face.FaceDetector> instances = new HashMap<>();

  FaceDetectorOptions faceDetectorOptions =
    new FaceDetectorOptions.Builder()
      .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
      .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
      .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
      .build();

  private void setErrorResult(String error) {
    resultMap.put("status", FaceDetectionStatus.ERROR.name().toLowerCase());
    resultMap.put("faceDirection", Utils.convertKebabCase(FaceDirection.UNKNOWN));
    resultMap.put("error", error);
    resultMap.put("frameData", null);
    resultMap.put("faces", null);
  }

  VisionCameraFaceDetectorPlugin(@Nullable Map<String, Object> options) {
    super(options);
    Log.e("zaaaaosman params", options == null ? "osman" : options.toString());
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
      FaceDetector faceDetector = instances.get(id);

      if (faceDetector == null) {
        faceDetector = FaceDetection.getClient(faceDetectorOptions);
        instances.put(id, faceDetector);
      }

      InputImage image = InputImage.fromMediaImage(mediaImage, Utils.convertRotationDegreeFromString(frame.getOrientation()));

      Task<List<Face>> task = faceDetector.process(image).addOnSuccessListener(
        visionFaces -> {
          List<Map<String, Object>> faces = new ArrayList<>(visionFaces.size());
          if (visionFaces.isEmpty()) {
            setErrorResult(new FaceDetectorException(104, "Faces not found").toString());
          }
          for (Face face : visionFaces) {

            Map<String, Object> faceData = new HashMap<>();

            Map<String, Integer> frameData = new HashMap<>();
            Rect rect = face.getBoundingBox();
            frameData.put("left", rect.left);
            frameData.put("top", rect.top);
            frameData.put("right", rect.right);
            frameData.put("bottom", rect.bottom);
            faceData.put("rect", frameData);

            faceData.put("headEulerAngleX", face.getHeadEulerAngleX());
            faceData.put("headEulerAngleY", face.getHeadEulerAngleY());
            faceData.put("headEulerAngleZ", face.getHeadEulerAngleZ());

            if (face.getSmilingProbability() != null) {
              faceData.put("smilingProbability", face.getSmilingProbability());
            }

            if (face.getLeftEyeOpenProbability()
              != null) {
              faceData.put("leftEyeOpenProbability", face.getLeftEyeOpenProbability());
            }

            if (face.getRightEyeOpenProbability()
              != null) {
              faceData.put("rightEyeOpenProbability", face.getRightEyeOpenProbability());
            }

            if (face.getTrackingId() != null) {
              faceData.put("trackingId", face.getTrackingId());
            }
            faceData.put("landmarks", getLandmarkData(face));

            faceData.put("contours", getContourData(face));

            faces.add(faceData);
          }
          resultMap.put("faces", faces.toString());

        }).addOnFailureListener(
        (Exception e) -> {
          Log.e("FaceDetection", e.getMessage());
          setErrorResult(e.toString());
        }
      );


      //Convert frame bitmap to base64
      Bitmap frameInBitmap = Utils.convertImageToBitmap(image);
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      frameInBitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);

      String frameInBase64 = Base64.encodeToString(baos.toByteArray(), Base64.DEFAULT);

      resultMap.put("status", FaceDetectionStatus.SUCCESS.name().toLowerCase());
      resultMap.put("frameData", frameInBase64);
      return resultMap;

    } catch (FaceDetectorException e) {
      Log.e("FaceDetection", e.getMessage());
      setErrorResult(e.toString());
      return resultMap;
    } catch (Exception e) {
      Log.e("FaceDetection", e.toString());
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

  private double[] landmarkPosition(Face face, int landmarkInt) {
    FaceLandmark landmark = face.getLandmark(landmarkInt);
    if (landmark != null) {
      return new double[]{landmark.getPosition().x, landmark.getPosition().y};
    }
    return null;
  }

  private Map<String, double[]> getLandmarkData(Face face) {
    Map<String, double[]> landmarks = new HashMap<>();

    landmarks.put("bottomMouth", landmarkPosition(face, FaceLandmark.MOUTH_BOTTOM));
    landmarks.put("rightMouth", landmarkPosition(face, FaceLandmark.MOUTH_RIGHT));
    landmarks.put("leftMouth", landmarkPosition(face, FaceLandmark.MOUTH_LEFT));
    landmarks.put("rightEye", landmarkPosition(face, FaceLandmark.RIGHT_EYE));
    landmarks.put("leftEye", landmarkPosition(face, FaceLandmark.LEFT_EYE));
    landmarks.put("rightEar", landmarkPosition(face, FaceLandmark.RIGHT_EAR));
    landmarks.put("leftEar", landmarkPosition(face, FaceLandmark.LEFT_EAR));
    landmarks.put("rightCheek", landmarkPosition(face, FaceLandmark.RIGHT_CHEEK));
    landmarks.put("leftCheek", landmarkPosition(face, FaceLandmark.LEFT_CHEEK));
    landmarks.put("noseBase", landmarkPosition(face, FaceLandmark.NOSE_BASE));

    return landmarks;
  }

  private Map<String, List<double[]>> getContourData(Face face) {
    Map<String, List<double[]>> contours = new HashMap<>();

    contours.put("face", contourPosition(face, FaceContour.FACE));
    contours.put(
      "leftEyebrowTop", contourPosition(face, FaceContour.LEFT_EYEBROW_TOP));
    contours.put(
      "leftEyebrowBottom", contourPosition(face, FaceContour.LEFT_EYEBROW_BOTTOM));
    contours.put(
      "rightEyebrowTop", contourPosition(face, FaceContour.RIGHT_EYEBROW_TOP));
    contours.put(
      "rightEyebrowBottom",
      contourPosition(face, FaceContour.RIGHT_EYEBROW_BOTTOM));
    contours.put("leftEye", contourPosition(face, FaceContour.LEFT_EYE));
    contours.put("rightEye", contourPosition(face, FaceContour.RIGHT_EYE));
    contours.put("upperLipTop", contourPosition(face, FaceContour.UPPER_LIP_TOP));
    contours.put(
      "upperLipBottom", contourPosition(face, FaceContour.UPPER_LIP_BOTTOM));
    contours.put("lowerLipTop", contourPosition(face, FaceContour.LOWER_LIP_TOP));
    contours.put(
      "lowerLipBottom", contourPosition(face, FaceContour.LOWER_LIP_BOTTOM));
    contours.put("noseBridge", contourPosition(face, FaceContour.NOSE_BRIDGE));
    contours.put("noseBottom", contourPosition(face, FaceContour.NOSE_BOTTOM));
    contours.put("leftCheek", contourPosition(face, FaceContour.LEFT_CHEEK));
    contours.put("rightCheek", contourPosition(face, FaceContour.RIGHT_CHEEK));

    return contours;
  }

  private List<double[]> contourPosition(Face face, int contourInt) {
    FaceContour contour = face.getContour(contourInt);
    if (contour != null) {
      List<PointF> contourPoints = contour.getPoints();
      List<double[]> result = new ArrayList<>();
      for (int i = 0; i < contourPoints.size(); i++) {
        result.add(new double[]{contourPoints.get(i).x, contourPoints.get(i).y});
      }
      return result;
    }
    return null;
  }


}
