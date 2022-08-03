#include "NestModel.hpp"

NestModel::NestModel(const Shared* const sh, const uint64_t size) : shared(sh), cm(sh, size, nCM, 64) {}

void NestModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_c1
    int c = c1;
    int matched = 1;
    int vv = 0;
    w *= static_cast<int>((vc & 7) > 0 && (vc & 7) < 3);
    if((c & 0x80) != 0 ) {
      w = w * 11 * 32 + c;
    }
    const int lc = (c >= 'A' && c <= 'Z' ? c + 'a' - 'A' : c);
    if( lc == 'a' || lc == 'e' || lc == 'i' || lc == 'o' || lc == 'u' ) {
      vv = 1;
      w = w * 997 * 8 + (lc / 4 - 22);
    } else if( lc >= 'a' && lc <= 'z' ) {
      vv = 2;
      w = w * 271 * 32 + lc - 97;
    } else if( lc == ' ' || lc == '.' || lc == ',' || lc == '\n' ) {
      vv = 3;
    } else if( lc >= '0' && lc <= '9' ) {
      vv = 4;
    } else if( lc == 'y' ) {
      vv = 5;
    } else if( lc == '\'' ) {
      vv = 6;
    } else {
      vv = (c & 32) != 0 ? 7 : 0;
    }
    vc = (vc << 3) | vv;
    if( vv != lvc ) {
      wc = (wc << 3) | vv;
      lvc = vv;
    }
    INJECT_SHARED_c4
    switch( c ) {
      case ' ':
        qc = 0;
        break;
      case '(':
        ic += 31;
        break;
      case ')':
        ic -= 31;
        break;
      case '[':
        ic += 11;
        break;
      case ']':
        ic -= 11;
        break;
      case '<':
        ic += 23;
        qc += 34;
        break;
      case '>':
        ic -= 23;
        qc /= 5;
        break;
      case ':':
        pc = 20;
        break;
      case '{':
        ic += 17;
        break;
      case '}':
        ic -= 17;
        break;
      case '|':
        pc += 223;
        break;
      case '"':
        pc += 0x40;
        break;
      case '\'':
        pc += 0x42;
        if( c != static_cast<uint8_t>(c4 >> 8)) {
          sense2 ^= 1;
        } else {
          ac += (2 * sense2 - 1);
        }
        break;
      case '\n':
        pc = qc = 0;
        break;
      case '.':
      case '!':
      case '?':
        pc = 0;
        break;
      case '#':
        pc += 0x08;
        break;
      case '%':
        pc += 0x76;
        break;
      case '$':
        pc += 0x45;
        break;
      case '*':
        pc += 0x35;
        break;
      case '-':
        pc += 0x3;
        break;
      case '@':
        pc += 0x72;
        break;
      case '&':
        qc += 0x12;
        break;
      case ';':
        qc /= 3;
        break;
      case '\\':
        pc += 0x29;
        break;
      case '/':
        pc += 0x11;
        if( c1 == '<' ) {
          qc += 74;
        }
        break;
      case '=':
        pc += 87;
        if( c != static_cast<uint8_t>(c4 >> 8)) {
          sense1 ^= 1;
        } else {
          ec += (2 * sense1 - 1);
        }
        break;
      default:
        matched = 0;
    }
    if( c4 == 0x266C743B ) {
      uc = min(7, uc + 1); //&lt;
    } else if( c4 == 0x2667743B ) {
      uc -= static_cast<int>(uc > 0); //&gt;
    }
    if( matched != 0 ) {
      bc = 0;
    } else {
      bc += 1;
    }
    if( bc > 300 ) {
      bc = ic = pc = qc = uc = 0;
    }
    const uint8_t R_ = CM_USE_RUN_STATS;
    uint64_t i = 0;
    cm.set(R_, hash(++i, (vv > 0 && vv < 3) ? 0 : (lc | 0x100), ic & 0x03FF, ec & 0x07, ac & 0x07, uc));
    cm.set(R_, hash(++i, ic, w, ilog2(bc + 1)));
    cm.set(R_, hash(++i, (3 * vc + 77 * pc + 373 * ic + qc) & 0xffff));
    cm.set(R_, hash(++i, (31 * vc + 27 * pc + 281 * qc) & 0xffff));
    cm.set(R_, hash(++i, (13 * vc + 271 * ic + qc + bc) & 0xffff));
    cm.set(R_, hash(++i, (17 * pc + 7 * ic) & 0xffff));
    cm.set(R_, hash(++i, (13 * vc + ic) & 0xffff));
    cm.set(R_, hash(++i, (vc / 3 + pc) & 0xffff));
    cm.set(R_, hash(++i, (7 * wc + qc) & 0xffff));
    cm.set(R_, hash(++i, vc & 0xffff, c4 & 0xff));
    cm.set(R_, hash(++i, (3 * pc) & 0xffff, c4 & 0xff));
    cm.set(R_, hash(++i, ic & 0xffff, c4 & 0xff));
  }
  cm.mix(m);
}
