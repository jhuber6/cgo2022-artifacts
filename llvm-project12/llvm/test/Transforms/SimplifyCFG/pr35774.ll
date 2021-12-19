; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S %s | FileCheck %s

%foo = type { i32 (%foo)*, i32 }

declare i32 @putchar(i32)

define i32 @intercept(%foo %f) {
; CHECK-LABEL: @intercept(
; CHECK-NEXT:    [[FN:%.*]] = extractvalue [[FOO:%.*]] %f, 0
; CHECK-NEXT:    [[X:%.*]] = extractvalue [[FOO]] %f, 1
; CHECK-NEXT:    [[X0:%.*]] = icmp eq i32 [[X]], 0
; CHECK-NEXT:    br i1 [[X0]], label [[ZERO:%.*]], label [[NONZERO:%.*]]
; CHECK:       Zero:
; CHECK-NEXT:    [[R0:%.*]] = musttail call i32 [[FN]](%foo [[F:%.*]])
; CHECK-NEXT:    ret i32 [[R0]]
; CHECK:       Nonzero:
; CHECK-NEXT:    [[R1:%.*]] = tail call i32 [[FN]](%foo [[F]])
; CHECK-NEXT:    [[TMP1:%.*]] = tail call i32 @putchar(i32 [[R1]])
; CHECK-NEXT:    ret i32 [[R1]]
;
  %fn = extractvalue %foo %f, 0
  %x = extractvalue %foo %f, 1
  %x0 = icmp eq i32 %x, 0
  br i1 %x0, label %Zero, label %Nonzero

Zero:
  %r0 = musttail call i32 %fn(%foo %f)
  ret i32 %r0

Nonzero:
  %r1 = tail call i32 %fn(%foo %f)
  %1 = tail call i32 @putchar(i32 %r1)
  ret i32 %r1
}