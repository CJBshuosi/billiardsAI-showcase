import 'package:flutter/material.dart';

void main() {
  runApp(const BilliardsAiApp());
}

class BilliardsAiApp extends StatelessWidget {
  const BilliardsAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Billiards AI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
      ),
      home: const LiveCoachPlaceholderPage(),
    );
  }
}

class LiveCoachPlaceholderPage extends StatelessWidget {
  const LiveCoachPlaceholderPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Billiards AI')),
      body: const SafeArea(
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('实时教练准备中'),
              SizedBox(height: 8),
              Text('尚未连接视频或教练事件'),
            ],
          ),
        ),
      ),
    );
  }
}
