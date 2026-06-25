// main.dart
// Camera feed, YOLO inference, ReID tracking, and overlay UI.
// Compatible with: Flutter 3.44.4, Dart 3.8.x, camera ^0.10.5+9
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'yolo_detector.dart';

// ─────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────
void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final cameras = await availableCameras();

  runApp(MyApp(camera: cameras.first));
}

// ─────────────────────────────────────────────
// Root app
// ─────────────────────────────────────────────
class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({
    super.key,
    required this.camera,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Object Detection',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: TrackerScreen(camera: camera),
    );
  }
}

// ─────────────────────────────────────────────
// Main Screen
// ─────────────────────────────────────────────
class TrackerScreen extends StatefulWidget {
  final CameraDescription camera;

  const TrackerScreen({
    super.key,
    required this.camera,
  });

  @override
  State<TrackerScreen> createState() => _TrackerScreenState();
}

class _TrackerScreenState extends State<TrackerScreen> {
  //────────────────────────────────────────────
  // Components
  //────────────────────────────────────────────
  late CameraController _cameraController;

  final YoloDetector _detector = YoloDetector();

  //────────────────────────────────────────────
  // State
  //────────────────────────────────────────────
  bool _isInitialized = false;
  bool _isProcessing = false;

  // Latest YOLO detections
  List<Detection> _detections = [];

  //────────────────────────────────────────────
  // FPS Counter
  //────────────────────────────────────────────
  int _fps = 0;
  int _frameCount = 0;

  DateTime _lastFpsTime = DateTime.now();

  //────────────────────────────────────────────
  // Camera preview size
  //────────────────────────────────────────────
  Size _previewSize = const Size(1, 1);

  //────────────────────────────────────────────
  // Lifecycle
  //────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _detector.dispose();
    super.dispose();
  }

  // ─────────────────────────────────────────────
// Camera + Model Initialization
// ─────────────────────────────────────────────
Future<void> _initCamera() async {
  _cameraController = CameraController(
    widget.camera,
    ResolutionPreset.low,
    enableAudio: false,
    imageFormatGroup: ImageFormatGroup.yuv420,
  );

  await _cameraController.initialize();

  // Load YOLO model
  await _detector.init();

  // Store preview size
  final previewSize = _cameraController.value.previewSize!;
  _previewSize = Size(
    previewSize.height,
    previewSize.width,
  );

  // Start camera stream
  await _cameraController.startImageStream(_onFrame);

  if (mounted) {
    setState(() {
      _isInitialized = true;
    });
  }
}

// ─────────────────────────────────────────────
// Called for every camera frame
// ─────────────────────────────────────────────
void _onFrame(CameraImage cameraImage) {
  // Skip frame if previous inference is still running
  if (_isProcessing) return;

  _isProcessing = true;

  _processFrame(cameraImage).whenComplete(() {
    _isProcessing = false;
  });
}

// ─────────────────────────────────────────────
// Process one frame
// ─────────────────────────────────────────────
Future<void> _processFrame(CameraImage cameraImage) async {
  try {
    final t0 = DateTime.now();

    // Convert camera frame to RGB image
    final frame = _convertYUV420(cameraImage);

    if (frame == null) return;

    final t1 = DateTime.now();

    // Run YOLO
    final detections = await _detector.detect(frame);

    final t2 = DateTime.now();

    print(
      "Convert: ${t1.difference(t0).inMilliseconds} ms | "
      "Detect: ${t2.difference(t1).inMilliseconds} ms",
    );

    // FPS Counter
    _frameCount++;

    final now = DateTime.now();

    if (now.difference(_lastFpsTime).inSeconds >= 1) {
      _fps = _frameCount;
      _frameCount = 0;
      _lastFpsTime = now;
    }

    if (mounted) {
      setState(() {
        _detections = detections;
      });
    }
  } catch (e) {
    debugPrint(e.toString());
  }
}
 // ─────────────────────────────────────────────
// Convert Android YUV420 CameraImage → RGB Image
// ─────────────────────────────────────────────
img.Image? _convertYUV420(CameraImage image) {
  try {
    final width = image.width;
    final height = image.height;

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final yBytes = yPlane.bytes;
    final uBytes = uPlane.bytes;
    final vBytes = vPlane.bytes;

    final yRowStride = yPlane.bytesPerRow;
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    final output = img.Image(
      width: width,
      height: height,
    );

    for (int y = 0; y < height; y++) {
      final yRow = y * yRowStride;
      final uvRow = (y ~/ 2) * uvRowStride;

      for (int x = 0; x < width; x++) {
        final yIndex = yRow + x;
        final uvIndex = uvRow + (x ~/ 2) * uvPixelStride;

        if (yIndex >= yBytes.length ||
            uvIndex >= uBytes.length ||
            uvIndex >= vBytes.length) {
          continue;
        }

        final yValue = yBytes[yIndex];
        final uValue = uBytes[uvIndex] - 128;
        final vValue = vBytes[uvIndex] - 128;

        final r =
            (yValue + 1.370705 * vValue).round().clamp(0, 255);

        final g =
            (yValue -
                    0.337633 * uValue -
                    0.698001 * vValue)
                .round()
                .clamp(0, 255);

        final b =
            (yValue + 1.732446 * uValue).round().clamp(0, 255);

        output.setPixelRgb(x, y, r, g, b);
      }
    }

    final rotated = img.copyRotate(output, angle: 90);
    return rotated;
  } catch (e) {
    debugPrint("YUV conversion failed: $e");
    return null;
  }
}
  // ─────────────────────────────────────────────
// Build UI
// ─────────────────────────────────────────────
@override
Widget build(BuildContext context) {
  if (!_isInitialized) {
    return const Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(
              color: Colors.greenAccent,
            ),
            SizedBox(height: 16),
            Text(
              'Loading YOLO model...',
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
              ),
            ),
          ],
        ),
      ),
    );
  }

  return Scaffold(
    backgroundColor: Colors.black,
    body: Stack(
      children: [
        // Camera Preview
        Positioned.fill(
          child: CameraPreview(_cameraController),
        ),

        // Bounding Boxes
        Positioned.fill(
          child: CustomPaint(
            painter: BoundingBoxPainter(
              detections: _detections,
              previewSize: _previewSize,
            ),
          ),
        ),

        // FPS Panel
        Positioned(
          top: 48,
          right: 12,
          child: Container(
            padding: const EdgeInsets.symmetric(
              horizontal: 14,
              vertical: 10,
            ),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.55),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(
              'FPS : $_fps',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
      ],
    ),
  );
}
}

// ─────────────────────────────────────────────
// Bounding Box Painter
// ─────────────────────────────────────────────
class BoundingBoxPainter extends CustomPainter {
  final List<Detection> detections;
  final Size previewSize;

  BoundingBoxPainter({
    required this.detections,
    required this.previewSize,
  });

  static const Map<int, String> classNames = {
    0: 'Bottle',
    1: 'Mug',
    2: 'Remote',
  };
   @override
  void paint(Canvas canvas, Size size) {
    // Scale detection coordinates to screen coordinates
    final double scaleX =
        previewSize.width > 0 ? size.width / previewSize.width : 1.0;

    final double scaleY =
        previewSize.height > 0 ? size.height / previewSize.height : 1.0;

    final boxPaint = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke;

    for (final det in detections) {
      final left = det.x1 * scaleX;
      final top = det.y1 * scaleY;
      final right = det.x2 * scaleX;
      final bottom = det.y2 * scaleY;

      // Draw bounding box
      canvas.drawRect(
        Rect.fromLTRB(left, top, right, bottom),
        boxPaint,
      );

      // Class label with confidence
      final label =
          '${classNames[det.classId] ?? "Unknown"} ${(det.confidence * 100).toStringAsFixed(1)}%';

      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.bold,
            backgroundColor: Colors.black54,
          ),
        ),
        textDirection: TextDirection.ltr,
      );

      textPainter.layout();

      final labelY =
          (top - textPainter.height - 4).clamp(0.0, size.height);

      textPainter.paint(
        canvas,
        Offset(left, labelY),
      );
    }
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
