#include "XMLModel.hpp"
#include "../CharacterNames.hpp"

void XMLModel::detectContent(XMLContent *content) {
  INJECT_SHARED_buf
  INJECT_SHARED_c4
  INJECT_SHARED_c8
  if((c4 & 0xF0F0F0F0) == 0x30303030 ) { //may be 4 digits (dddd)
    int i = 0;
    int j = 0;
    while((i < 4) && ((j = (c4 >> (8 * i)) & 0xFF) >= 0x30 && j <= 0x39)) {
      i++;
    }
    if( i == 4 /*????dddd*/ && (((c8 & 0xFDF0F0FD) == 0x2D30302D && buf(9) >= 0x30 && buf(9) <= 0x39 /*d-dd-dddd or d.dd.dddd*/) ||
                                ((c8 & 0xF0FDF0FD) == 0x302D302D) /*d-d-dddd or d.d.dddd*/)) {
      (*content).type |= ContentFlags::Date;
    }
  } else if(((c8 & 0xF0F0FDF0) == 0x30302D30 /*dd.d???? or dd-d????*/ || (c8 & 0xF0F0F0FD) == 0x3030302D) &&
            buf(9) >= 0x30 && buf(9) <= 0x39 /*dddd-???? or dddd.????*/) {
    int i = 2;
    int j = 0;
    while((i < 4) && ((j = (c8 >> (8 * i)) & 0xFF) >= 0x30 && j <= 0x39)) {
      i++;
    }

    if( i == 4 && (c4 & 0xF0FDF0F0) == 0x302D3030 ) { //dd??d.dd or dd??d-dd
      (*content).type |= ContentFlags::Date;
    }
  }

  if((c4 & 0xF0FFF0F0) == 0x303A3030 && buf(5) >= 0x30 && buf(5) <= 0x39 && ((buf(6) < 0x30 || buf(6) > 0x39) /*?dd:dd*/ ||
                                                                                     ((c8 & 0xF0F0FF00) == 0x30303A00 &&
                                                                                      (buf(9) < 0x30 || buf(9) > 0x39) /*?dd:dd:dd*/))) {
    (*content).type |= ContentFlags::Time;
  }

  if((*content).length >= 8 && (c8 & 0x80808080) == 0 && (c4 & 0x80808080) == 0 ) { //8 ~ascii
    (*content).type |= ContentFlags::Text;
  }

  if((c8 & 0xF0F0FF) == 0x3030C2 && (c4 & 0xFFF0F0FF) == 0xB0303027 ) { //dd {utf8 C2B0: degree sign} dd {apostrophe}
    int i = 2;
    while((i < 7) && buf(i) >= 0x30 && buf(i) <= 0x39 ) {
      i += (i & 1) * 2 + 1;
    }

    if( i == 10 ) {
      (*content).type |= ContentFlags::Coordinates;
    }
  }

  if((c4 & 0xFFFFFA) == 0xC2B042 && (c4 & 0xff) != 0x47 &&
     (((c4 >> 24) >= 0x30 && (c4 >> 24) <= 0x39) ||
      ((c4 >> 24) == 0x20 && (buf(5) >= 0x30 && buf(5) <= 0x39)))) {
    (*content).type |= ContentFlags::Temperature;
  }

  INJECT_SHARED_c1
  if( c1 >= 0x30 && c1 <= 0x39 ) {
    (*content).type |= ContentFlags::Number;
  }

  if( c4 == 0x4953424E && (c8 & 0xff) == 0x20 ) { // " ISBN"
    (*content).type |= ContentFlags::ISBN;
  }
}

XMLModel::XMLModel(const Shared* const sh, const uint64_t size) : shared(sh), cm(sh, size, nCM, 64) {}

void XMLModel::update() {
  XMLTag *pTag = &cache.tags[(cache.Index - 1) & (cacheSize - 1)];
  XMLTag *tag = &cache.tags[cache.Index & (cacheSize - 1)];
  XMLAttribute *attribute = &((*tag).attributes.items[(*tag).attributes.Index & 3]);
  XMLContent *content = &(*tag).content;
  pState = state;
  INJECT_SHARED_c1
  INJECT_SHARED_c4
  if((c1 == TAB || c1 == SPACE) && (c1 == static_cast<uint8_t>(c4 >> 8) || (whiteSpaceRun == 0))) {
    whiteSpaceRun++;
    indentTab = static_cast<uint32_t>(c1 == TAB);
  } else {
    if((state == None || (state == ReadContent && (*content).length <= lineEnding + whiteSpaceRun)) && whiteSpaceRun > 1 + indentTab &&
       whiteSpaceRun != pWsRun ) {
      indentStep = abs(static_cast<int>(whiteSpaceRun - pWsRun));
      pWsRun = whiteSpaceRun;
    }
    whiteSpaceRun = 0;
  }
  if( c1 == NEW_LINE ) {
    lineEnding = 1 + (static_cast<uint8_t>(c4 >> 8) == CARRIAGE_RETURN);
  }

  const uint8_t R_ = CM_USE_RUN_STATS;

  INJECT_SHARED_c8
  switch( state ) {
    case None: {
      if( c1 == 0x3C ) {
        state = ReadTagName;
        memset(tag, 0, sizeof(XMLTag));
        (*tag).level = ((*pTag).endTag || (*pTag).empty) ? (*pTag).level : (*pTag).level + 1;
      }
      if((*tag).level > 1 ) {
        detectContent(content);
      }

      cm.set(R_, hash(pState, state, ((*pTag).level + 1) * indentStep - whiteSpaceRun));
      break;
    }
    case ReadTagName: {
      if((*tag).length > 0 && (c1 == TAB || c1 == NEW_LINE || c1 == CARRIAGE_RETURN || c1 == SPACE)) {
        state = ReadTag;
      } else if((c1 == 0x3A || (c1 >= 'A' && c1 <= 'Z') || c1 == 0x5F ||
                 (c1 >= 'a' && c1 <= 'z')) ||
                ((*tag).length > 0 && (c1 == 0x2D || c1 == 0x2E || (c1 >= '0' && c1 <= '9')))) {
        (*tag).length++;
        (*tag).name = (*tag).name * 263 * 32 + (c1 & 0xDF);
      } else if( c1 == 0x3E ) {
        if((*tag).endTag ) {
          state = None;
          cache.Index++;
        } else {
          state = ReadContent;
        }
      } else if( c1 != 0x21 && c1 != 0x2D && c1 != 0x2F && c1 != 0x5B ) {
        state = None;
        cache.Index++;
      } else if((*tag).length == 0 ) {
        if( c1 == 0x2F ) {
          (*tag).endTag = true;
          (*tag).level = max(0, (*tag).level - 1);
        } else if( c4 == 0x3C212D2D ) {
          state = ReadComment;
          (*tag).level = max(0, (*tag).level - 1);
        }
      }

      if((*tag).length == 1 && (c4 & 0xFFFF00) == 0x3C2100 ) {
        memset(tag, 0, sizeof(XMLTag));
        state = None;
      } else if((*tag).length == 5 && c8 == 0x215B4344 && c4 == 0x4154415B ) {
        state = ReadCDATA;
        (*tag).level = max(0, (*tag).level - 1);
      }

      int i = 1;
      do {
        pTag = &cache.tags[(cache.Index - i) & (cacheSize - 1)];
        i += 1 + static_cast<int>((*pTag).endTag && cache.tags[(cache.Index - i - 1) & (cacheSize - 1)].name == (*pTag).name);
      } while( i < cacheSize && ((*pTag).endTag || (*pTag).empty));

      cm.set(R_, hash(pState, state, (*tag).name, (*tag).level, (*pTag).name, static_cast<uint64_t>((*pTag).level != (*tag).level)));
      break;
    }
    case ReadTag: {
      if( c1 == 0x2F ) {
        (*tag).empty = true;
      } else if( c1 == 0x3E ) {
        if((*tag).empty ) {
          state = None;
          cache.Index++;
        } else {
          state = ReadContent;
        }
      } else if( c1 != TAB && c1 != NEW_LINE && c1 != CARRIAGE_RETURN && c1 != SPACE ) {
        state = ReadAttributeName;
        (*attribute).name = c1 & 0xDF;
      }
      cm.set(R_, hash(pState, state, (*tag).name, c1, (*tag).attributes.Index));
      break;
    }
    case ReadAttributeName: {
      if((c4 & 0xFFF0) == 0x3D20 && (c1 == 0x22 || c1 == 0x27)) {
        state = ReadAttributeValue;
        if((c8 & 0xDFDF) == 0x4852 && (c4 & 0xDFDF0000) == 0x45460000 ) {
          (*content).type |= Link;
        }
      } else if( c1 != 0x22 && c1 != 0x27 && c1 != 0x3D ) {
        (*attribute).name = (*attribute).name * 263 * 32 + (c1 & 0xDF);
      }

      cm.set(R_, hash(pState, state, (*attribute).name, (*tag).attributes.Index, (*tag).name, (*content).type));
      break;
    }
    case ReadAttributeValue: {
      if( c1 == 0x22 || c1 == 0x27 ) {
        (*tag).attributes.Index++;
        state = ReadTag;
      } else {
        (*attribute).value = (*attribute).value * 263 * 32 + (c1 & 0xDF);
        (*attribute).length++;
        if((c8 & 0xDFDFDFDF) == 0x48545450 && ((c4 >> 8) == 0x3A2F2F || c4 == 0x733A2F2F)) {
          (*content).type |= URL;
        }
      }
      cm.set(R_, hash(pState, state, (*attribute).name, (*content).type));
      break;
    }
    case ReadContent: {
      if( c1 == 0x3C ) {
        state = ReadTagName;
        cache.Index++;
        memset(&cache.tags[cache.Index & (cacheSize - 1)], 0, sizeof(XMLTag));
        cache.tags[cache.Index & (cacheSize - 1)].level = (*tag).level + 1;
      } else {
        (*content).length++;
        (*content).data = (*content).data * 997 * 16 + (c1 & 0xDF);
        detectContent(content);
      }
      cm.set(R_, hash(pState, state, (*tag).name, c4 & 0xC0FF));
      break;
    }
    case ReadCDATA: {
      if((c4 & 0xFFFFFF) == 0x5D5D3E ) {
        state = None;
        cache.Index++;
      }
      cm.set(R_, hash(pState, state));
      break;
    }
    case ReadComment: {
      if((c4 & 0xFFFFFF) == 0x2D2D3E ) {
        state = None;
        cache.Index++;
      }
      cm.set(R_, hash(pState, state));
      break;
    }
  }

  stateBh[pState] = (stateBh[pState] << 8) | c1;
  pTag = &cache.tags[(cache.Index - 1) & (cacheSize - 1)];

  uint64_t i = 64;
  cm.set(R_, hash(++i, state, (*tag).level, pState * 2 + static_cast<int>((*tag).endTag), (*tag).name));
  cm.set(R_, hash(++i, (*pTag).name, state * 2 + static_cast<int>((*pTag).endTag), (*pTag).content.type, (*tag).content.type));
  cm.set(R_, hash(++i, state * 2 + static_cast<int>((*tag).endTag), (*tag).name, (*tag).content.type, c4 & 0xE0FF));
}

void XMLModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    update();
  }
  cm.mix(m);
}
