/* eslint-disable react-hooks/exhaustive-deps */
import React, { useCallback, useEffect, useState } from 'react';
import { StyleSheet, Text, SafeAreaView, View, Image } from 'react-native';
import {
  Camera,
  runAtTargetFps,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';

import {
  FaceDetectionStatus,
  FaceDetectorResponse,
  FaceDetectorStatus,
  detectFace,
} from 'vision-camera-face-detector-plugin';

const CAMERA_SIZE = 250;

export default function App() {
  const [errMessage, setErrMessage] = useState<string>('');
  const [base64Frame, setBase64Frame] = useState<string>('');
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [status, setStatus] = useState<FaceDetectionStatus>('success');
  const [detectorStatus, setDetectorStatus] = useState<FaceDetectorStatus>();
  const id = (Date.now() * 1000).toString();
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();


  const startFaceDetector = () => {
    setDetectorStatus("startFaceDetector")
  }
  const stopFaceDetector = () => {
    setDetectorStatus("closeFaceDetector")
  }
  const submitSever = (frameData: string) => {
    setSubmitting(true);
    setErrMessage('');
    setBase64Frame(frameData);
    return new Promise((res) => {
      setTimeout(() => {
        setSubmitting(false);
        setBase64Frame('');
        res('data ne');
      }, 2000);
    });
  };

  const onGetFaceDetectorResponse = Worklets.createRunInJsFn(
    async (res: FaceDetectorResponse) => {
      setStatus(res.status);
      switch (res.status) {
        case 'error': {
          if (res.error) {
            setErrMessage(res.error.message);
          }
          break;
        }
        case 'success': {
          if (res.frameData) {
            console.log(res.faces);

            submitSever(res.frameData);
          }
          break;
        }
        default:
          break;
      }
    }
  );

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (submitting) {
        return;
      }
      runAtTargetFps(10, () => {
        'worklet'
        const response = detectFace(frame, detectorStatus, id);
        console.log(response);
        onGetFaceDetectorResponse(response);
      })
    },
    [submitting]
  );

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, []);

  const renderCamera = useCallback(() => {
    if (device == null || !hasPermission) {
      return <Text>No camera device</Text>;
    }

    return (
      <Camera
        device={device}
        isActive
        style={styles.camera}
        frameProcessor={frameProcessor}
        //ML Kit use YUV format
        pixelFormat="yuv"
      />
    );
  }, [device, hasPermission, frameProcessor]);

  const renderFrame = useCallback(() => {
    return (
      <Image
        style={{ width: 100, height: 100 }}
        source={{ uri: `data:image/png;base64,${base64Frame}` }}
      />
    );
  }, [base64Frame]);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.cameraContainer}>{renderCamera()}</View>
      {renderFrame()}
      <Text>{status === 'error' ? `Error: ${errMessage}` : status}</Text>
      
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },

  cameraContainer: {
    width: 450,
    height: 500,
    borderRadius: CAMERA_SIZE / 2,
    marginVertical: 24,
  },

  camera: {
    flex: 1,
  },
});
