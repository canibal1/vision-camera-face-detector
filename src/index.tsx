import { type Frame, VisionCameraProxy } from 'react-native-vision-camera';

export type FaceDirection =
  | 'left-skewed'
  | 'right-skewed'
  | 'frontal'
  | 'transitioning'
  | 'unknown';

/**
 *
 * @ErrorCode :
 * - 101: system error`
 * - 102: plugin not found
 * - 103: cannot get image from frame
 * - 104: faces not found
 * - 105: too many faces in frame
 * - 106: face is out of frame
 * - 107: face is transitioning
 */

type FaceDetectionErrorCode = 101 | 102 | 103 | 104 | 105 | 106 | 107;
export type FaceDetectionStatus = 'success' | 'standby' | 'error';
export type FaceDetectorStatus = 'startFaceDetector' | 'closeFaceDetector';
export type FaceDetectorResponse = {
  status: FaceDetectionStatus;
  faces: String;
  error?: {
    code: FaceDetectionErrorCode;
    message: string;
  };
};

const plugin = VisionCameraProxy.initFrameProcessorPlugin('detectFace', {});

export function detectFace(frame: Frame,status:FaceDetectorStatus,id:string): FaceDetectorResponse {
  'worklet';
  if (!plugin) {
    return {
      status: 'error',
      faces: 'unknown',
      error: {
        code: 102,
        message: 'Plugin not found',
      },
    };
  }
  //@ts-ignore
  return plugin.call(frame, { "id": id,"status":status });
}
