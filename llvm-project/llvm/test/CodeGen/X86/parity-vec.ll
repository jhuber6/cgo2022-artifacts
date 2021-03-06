; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+popcnt,+sse2 | FileCheck %s

define i1 @noncanonical_parity(<16 x i1> %x) {
; CHECK-LABEL: noncanonical_parity:
; CHECK:       # %bb.0:
; CHECK-NEXT:    psllw $7, %xmm0
; CHECK-NEXT:    pmovmskb %xmm0, %eax
; CHECK-NEXT:    popcntl %eax, %eax
; CHECK-NEXT:    andl $1, %eax
; CHECK-NEXT:    # kill: def $al killed $al killed $eax
; CHECK-NEXT:    retq
  %r = call i1 @llvm.vector.reduce.xor.v16i1(<16 x i1> %x)
  ret i1 %r
}
define i1 @canonical_parity(<16 x i1> %x) {
; CHECK-LABEL: canonical_parity:
; CHECK:       # %bb.0:
; CHECK-NEXT:    psllw $7, %xmm0
; CHECK-NEXT:    pmovmskb %xmm0, %eax
; CHECK-NEXT:    popcntw %ax, %ax
; CHECK-NEXT:    testb $1, %al
; CHECK-NEXT:    setne %al
; CHECK-NEXT:    retq
  %i1 = bitcast <16 x i1> %x to i16
  %i2 = call i16 @llvm.ctpop.i16(i16 %i1)
  %i3 = and i16 %i2, 1
  %i4 = icmp ne i16 %i3, 0
  ret i1 %i4
}
define i1 @canonical_parity_noncanonical_pred(<16 x i1> %x) {
; CHECK-LABEL: canonical_parity_noncanonical_pred:
; CHECK:       # %bb.0:
; CHECK-NEXT:    psllw $7, %xmm0
; CHECK-NEXT:    pmovmskb %xmm0, %eax
; CHECK-NEXT:    popcntw %ax, %ax
; CHECK-NEXT:    # kill: def $al killed $al killed $ax
; CHECK-NEXT:    retq
  %i1 = bitcast <16 x i1> %x to i16
  %i2 = call i16 @llvm.ctpop.i16(i16 %i1)
  %i3 = and i16 %i2, 1
  %i4 = icmp eq i16 %i3, 1
  ret i1 %i4
}

define i1 @noncanonical_nonparity(<16 x i1> %x) {
; CHECK-LABEL: noncanonical_nonparity:
; CHECK:       # %bb.0:
; CHECK-NEXT:    psllw $7, %xmm0
; CHECK-NEXT:    pmovmskb %xmm0, %eax
; CHECK-NEXT:    popcntl %eax, %eax
; CHECK-NEXT:    andl $1, %eax
; CHECK-NEXT:    xorb $1, %al
; CHECK-NEXT:    # kill: def $al killed $al killed $eax
; CHECK-NEXT:    retq
  %r.inv = call i1 @llvm.vector.reduce.xor.v16i1(<16 x i1> %x)
  %r = xor i1 %r.inv, -1
  ret i1 %r
}
define i1 @canonical_nonparity(<16 x i1> %x) {
; CHECK-LABEL: canonical_nonparity:
; CHECK:       # %bb.0:
; CHECK-NEXT:    psllw $7, %xmm0
; CHECK-NEXT:    pmovmskb %xmm0, %eax
; CHECK-NEXT:    popcntw %ax, %ax
; CHECK-NEXT:    testb $1, %al
; CHECK-NEXT:    sete %al
; CHECK-NEXT:    retq
  %i1 = bitcast <16 x i1> %x to i16
  %i2 = call i16 @llvm.ctpop.i16(i16 %i1)
  %i3 = and i16 %i2, 1
  %i4 = icmp eq i16 %i3, 0
  ret i1 %i4
}
define i1 @canonical_nonparity_noncanonical_pred(<16 x i1> %x) {
; CHECK-LABEL: canonical_nonparity_noncanonical_pred:
; CHECK:       # %bb.0:
; CHECK-NEXT:    psllw $7, %xmm0
; CHECK-NEXT:    pmovmskb %xmm0, %eax
; CHECK-NEXT:    popcntw %ax, %ax
; CHECK-NEXT:    andl $1, %eax
; CHECK-NEXT:    xorb $1, %al
; CHECK-NEXT:    # kill: def $al killed $al killed $eax
; CHECK-NEXT:    retq
  %i1 = bitcast <16 x i1> %x to i16
  %i2 = call i16 @llvm.ctpop.i16(i16 %i1)
  %i3 = and i16 %i2, 1
  %i4 = icmp ne i16 %i3, 1
  ret i1 %i4
}

declare i1 @llvm.vector.reduce.xor.v16i1(<16 x i1>)
declare i16 @llvm.ctpop.i16(i16)
