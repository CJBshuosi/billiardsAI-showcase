import 'package:billiards_ai_mobile/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows the product title and safe live-coach placeholder', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const BilliardsAiApp());

    expect(find.text('Billiards AI'), findsOneWidget);
    expect(find.text('实时教练准备中'), findsOneWidget);
    expect(find.text('尚未连接视频或教练事件'), findsOneWidget);
  });
}
