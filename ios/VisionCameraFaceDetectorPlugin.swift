import VisionCamera
import MLKitVision
import MLKitFaceDetection
import CoreMedia
import CoreImage


@objc(VisionCameraFaceDetectorPlugin)
public class VisionCameraFaceDetectorPlugin: FrameProcessorPlugin {
    let startFaceDetector = "startFaceDetector"
    let closeFaceDetector = "closeFaceDetector"
    
    private var instances = [String: FaceDetector]()
    
    var lastDetectionTime = 0
    
    
    
    let faceDetectorOptions = FaceDetectorOptions()
    
    
    var result: [String: Any] = ["status": "", "faceDirection": FaceDirection.unknown.rawValue, "error": FaceDetectionError(code: 102, message:"Plugin not found").toDictionary()]
    
    var _prevFaceDirection = FaceDirection.unknown
    
    var _firstDetectedTime = 0
    
    private func setErrorResult(errorCode: Int, errorMessage: String) {
        self.result = ["status": "error", "faceDirection": FaceDirection.unknown.rawValue, "error": FaceDetectionError(code: errorCode, message: errorMessage).toDictionary()]
    }
    
    class func newInstance() -> VisionCameraFaceDetectorPlugin {
        return VisionCameraFaceDetectorPlugin()
    }
    
    public override init() {
        super.init()
        faceDetectorOptions.performanceMode = .accurate
        faceDetectorOptions.landmarkMode = FaceDetectorLandmarkMode.all
        faceDetectorOptions.classificationMode = FaceDetectorClassificationMode.all
        faceDetectorOptions.isTrackingEnabled = true
        
    }
    
    
    public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable : Any]?) -> Any {
        
        
        if let status = arguments?["status"] as? String {
            
            switch status {
            case "startFaceDetector":
                handleDetection(frame, arguments:arguments)
            case "closeFaceDetector":
                if let args = arguments as? [String: Any], let uid = args["id"] as? String {
                    instances.removeValue(forKey: uid)
                }
            default:
                return result
            }
        }
            return result
       
    }
 //   func imageOrientation(
 //           deviceOrientation: UIDeviceOrientation,
 //           cameraPosition: AVCaptureDevice.Position
 //       ) -> UIImage.Orientation {
 //           switch deviceOrientation {
 //           case .portrait:
 //               return cameraPosition == .front ? .leftMirrored : .right
 //           case .landscapeLeft:
 //               return cameraPosition == .front ? .downMirrored : .up
 //           case .portraitUpsideDown:
 //               return cameraPosition == .front ? .rightMirrored : .left
 //           case .landscapeRight:
 //               return cameraPosition == .front ? .upMirrored : .down
 //           case .faceDown, .faceUp, .unknown:
 //               return .up
 //           }
 //       }

    private func handleDetection(_ frame: Frame, arguments: [AnyHashable : Any]?) -> Any{
        if let id = arguments?["id"]  as? String {
                // Status değerine göre switch-case yapısını kullanın
          
          
        let currentTime = Int(Date().timeIntervalSince1970 * 1000) // Convert to milliseconds
        
        if (currentTime - lastDetectionTime) < 1500 {
            return result
        }
        
        lastDetectionTime = currentTime

            var faceDetector = instances[id]
            if(faceDetector == nil)
            {
                faceDetector = FaceDetector.faceDetector(options: faceDetectorOptions)
                instances[id] = faceDetector
            }
            
        let buffer = frame.buffer
        let visionImage = VisionImage(buffer: buffer)
 //          visionImage.orientation = imageOrientation(
 //              deviceOrientation: UIDevice.current.orientation,
 //              cameraPosition: AVCaptureDevice.Position.front)
 //
      guard let pixelBuffer = CMSampleBufferGetImageBuffer(buffer) else {
          self.setErrorResult(errorCode: 102, errorMessage: "Cannot get image from frame") // cannot get image from frame
         return result
       }
       let frameWidth = CVPixelBufferGetWidth(pixelBuffer)
       let frameHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        weak var weakSelf = self
           faceDetector?.process(visionImage) { faces, error in
               guard weakSelf != nil else {
                   self.setErrorResult(errorCode: 101, errorMessage: "System error") // system error
                   return
               }
             
               guard let faces = faces, !faces.isEmpty else {
                   self.setErrorResult(errorCode: 104, errorMessage: "Face not found") // faces not found
                 return
               }
               
               guard faces.count == 1 else {
                 self.setErrorResult(errorCode: 105, errorMessage: "Too many faces in frame") // too many faces in frame
                 return
               }
               let userFace = faces.first
               
               //detect does face is fully visible in frame
               if(Utils.isFaceOutOfFrame(faceBoundingBox: FaceBoundingBox(rect: userFace!.frame), frameWidth: frameWidth , frameHeight: frameHeight)){
                   self.setErrorResult(errorCode: 106, errorMessage: "Face is out of frame") // face is out of frame
                   return
               }
               //convert frame to base64 image data
               guard let imageBuffer = buffer.imageBuffer else {
                 self.setErrorResult(errorCode: 102, errorMessage: "Cannot get image from frame") //cannot get image from frame
                 return
               }
               
               var facesData = [[String: Any]]() // Array to store dictionaries for each face

               for face in faces {
                   var faceData = [String: Any]()
                   
                   var frameData: [String:CGFloat] = [
                        "left": face.frame.origin.x,
                        "top": face.frame.origin.y,
                       "right":  face.frame.maxX,
                       "bottom":  face.frame.maxY
                   ]
                   faceData["rect"] = frameData

                   faceData["headEulerAngleX"] = face.headEulerAngleX
                   faceData["headEulerAngleY"] = face.headEulerAngleY
                   faceData["headEulerAngleZ"] = face.headEulerAngleZ
                   faceData["smilingProbability"] = face.smilingProbability
                   faceData["leftEyeOpenProbability"] = face.leftEyeOpenProbability
                   faceData["rightEyeOpenProbability"] = face.rightEyeOpenProbability
                   faceData["trackingId"] = face.trackingID
                   

                   

                   facesData.append(faceData)
               }

               
               let image = CIImage(cvPixelBuffer: imageBuffer)
               let context = CIContext()
               guard let cgImage = context.createCGImage(image, from: image.extent) else {
                 self.setErrorResult(errorCode: 102, errorMessage: "Cannot get image from frame") //cannot get image from frame
                 return
               }
               let uiImage = UIImage(cgImage: cgImage)
               let imageData = uiImage.jpegData(compressionQuality: 100)
               let frameData = imageData?.base64EncodedString()
           
               //set base64 FrameData to result
               self.result = ["status": "success",
                              "faces": facesData,
                              "frameData": frameData ?? ""]
         }
         return result
        
    
    } else {
        // Eğer 'data' veya 'status' yoksa veya beklenen türde değilse
        return result
    }

}
}
