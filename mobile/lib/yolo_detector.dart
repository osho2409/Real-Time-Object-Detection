// yolo_detector.dart
// YOLO11n inference via ONNX Runtime 1.4.1.
// Handles preprocessing, inference, output parsing, and NMS.
// Compatible with: onnxruntime ^1.4.1, image ^4.1.7, Dart 3.8.x

import 'dart:typed_data';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

//─────────────────────────────────────────────
// Detection object
//─────────────────────────────────────────────
class Detection {
  final int classId;
  final double confidence;
  final int x1;
  final int y1;
  final int x2;
  final int y2;

  Detection({
    required this.classId,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });
}

//─────────────────────────────────────────────
// YOLO11 ONNX Detector
// Flutter 3.x
// onnxruntime 1.4.1
//─────────────────────────────────────────────
class YoloDetector {
  OrtSession? _session;

  static const int inputSize = 320;

  // Slightly higher confidence reduces false detections
  static const double confThreshold = 0.70;

  static const double nmsThreshold = 0.45;
  static const int numClasses = 3;

  //───────────────────────────────────────────
  // Load ONNX model
  //───────────────────────────────────────────
  Future<void> init() async {
    OrtEnv.instance.init();

    final modelData = await rootBundle.load('assets/best.onnx');
    final modelBytes = modelData.buffer.asUint8List();

    final sessionOptions = OrtSessionOptions();

    // Enable graph optimizations
    sessionOptions.setSessionGraphOptimizationLevel(
      GraphOptimizationLevel.ortEnableAll,
    );

    // CPU thread configuration
    sessionOptions.setIntraOpNumThreads(2);
    sessionOptions.setInterOpNumThreads(1);

    // Use Android NNAPI with FP16 acceleration
    sessionOptions.appendNnapiProvider(
      NnapiFlags.useFp16,
    );

    _session = OrtSession.fromBuffer(
      modelBytes,
      sessionOptions,
    );
  }
 //─────────────────────────────────────────────
// Convert image to Float32 tensor (NCHW)
//─────────────────────────────────────────────
Float32List _preprocess(img.Image image) {
  final resized =
      (image.width == inputSize && image.height == inputSize)
          ? image
          : img.copyResize(
              image,
              width: inputSize,
              height: inputSize,
              interpolation: img.Interpolation.nearest,
            );

  final input = Float32List(3 * inputSize * inputSize);

  int idx = 0;

  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);

        switch (c) {
          case 0:
            input[idx++] = pixel.r / 255.0;
            break;

          case 1:
            input[idx++] = pixel.g / 255.0;
            break;

          case 2:
            input[idx++] = pixel.b / 255.0;
            break;
        }
      }
    }
  }

  return input;
}

//─────────────────────────────────────────────
// Run YOLO inference
//─────────────────────────────────────────────
Future<List<Detection>> detect(img.Image frame) async {
  if (_session == null) return [];

  final origW = frame.width;
  final origH = frame.height;

  final inputData = _preprocess(frame);

  final inputTensor = OrtValueTensor.createTensorWithDataList(
    inputData,
    [1, 3, inputSize, inputSize],
  );

  final feeds = <String, OrtValue>{
    'images': inputTensor,
  };

  final runOptions = OrtRunOptions();

  final outputs = await _session!.runAsync(
    runOptions,
    feeds,
  );

  inputTensor.release();
  runOptions.release();

  if (outputs == null || outputs.isEmpty) {
    return [];
  }

  final firstOutput = outputs.first;

  if (firstOutput == null) {
    return [];
  }

  final rawOutput = firstOutput.value;

  if (rawOutput == null) {
    firstOutput.release();
    return [];
  }

  final output = rawOutput as List<List<List<double>>>;

  final detections = _parseOutput(
    output.first,
    origW / inputSize,
    origH / inputSize,
    origW,
    origH,
  );

  for (final out in outputs) {
    out?.release();
  }

  return _nms(detections);
}
 // -------------------------------------------------------
// Parse YOLO output tensor
// -------------------------------------------------------
List<Detection> _parseOutput(
  List<List<double>> output,
  double scaleX,
  double scaleY,
  int origW,
  int origH,
) {
  final List<Detection> detections = [];

  final int numAnchors = output[0].length;

  for (int i = 0; i < numAnchors; i++) {
    double maxConf = 0.0;
    int classId = 0;

    // Find class with highest confidence
    for (int c = 0; c < numClasses; c++) {
      final conf = output[4 + c][i];

      if (conf > maxConf) {
        maxConf = conf;
        classId = c;
      }
    }

    // Ignore low-confidence detections
    if (maxConf < confThreshold) continue;

    final cx = output[0][i];
    final cy = output[1][i];
    final w = output[2][i];
    final h = output[3][i];

    final x1 = ((cx - w / 2) * scaleX).round().clamp(0, origW - 1);
    final y1 = ((cy - h / 2) * scaleY).round().clamp(0, origH - 1);
    final x2 = ((cx + w / 2) * scaleX).round().clamp(0, origW - 1);
    final y2 = ((cy + h / 2) * scaleY).round().clamp(0, origH - 1);

    // Invalid box
    if (x2 <= x1 || y2 <= y1) continue;

    // Ignore very tiny boxes (reduces false detections)
    final boxWidth = x2 - x1;
    final boxHeight = y2 - y1;

    if (boxWidth < 12 || boxHeight < 12) continue;

    detections.add(
      Detection(
        classId: classId,
        confidence: maxConf,
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
      ),
    );
  }

  return detections;
}

// -------------------------------------------------------
// Non-Maximum Suppression (Per Class)
// -------------------------------------------------------
List<Detection> _nms(List<Detection> detections) {
  detections.sort(
    (a, b) => b.confidence.compareTo(a.confidence),
  );

  final List<Detection> kept = [];

  for (final det in detections) {
    bool suppress = false;

    for (final existing in kept) {
      // Don't compare different classes
      if (det.classId != existing.classId) {
        continue;
      }

      if (_iou(det, existing) > nmsThreshold) {
        suppress = true;
        break;
      }
    }

    if (!suppress) {
      kept.add(det);
    }
  }

  return kept;
}
  // -------------------------------------------------------
// Intersection over Union (IoU)
// -------------------------------------------------------
double _iou(Detection a, Detection b) {
  final ix1 = max(a.x1, b.x1);
  final iy1 = max(a.y1, b.y1);
  final ix2 = min(a.x2, b.x2);
  final iy2 = min(a.y2, b.y2);

  if (ix2 <= ix1 || iy2 <= iy1) {
    return 0.0;
  }

  final intersection = (ix2 - ix1) * (iy2 - iy1);

  final areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  final areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

  final union = areaA + areaB - intersection;

  if (union <= 0) return 0.0;

  return intersection / union.toDouble();
}

// -------------------------------------------------------
// Cleanup
// -------------------------------------------------------
void dispose() {
  _session?.release();
  _session = null;
  OrtEnv.instance.release();
}
}