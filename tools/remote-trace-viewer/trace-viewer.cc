#include <SDL3/SDL.h>
#include <SDL3/SDL_render.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define INITIAL_WINDOW_WIDTH 1920
#define INITIAL_WINDOW_HEIGHT 1080
const float MINIMAP_Y = 8 + 12 + 8;
const float MINIMAP_HEIGHT = 100;
const float EVENTVIEW_EVENT_HEIGHT = 12;

#define STR(x) #x
#define ALICEBLUE 240, 248, 255
#define DARKGREY 169, 169, 169
#define GREEN 0, 128, 0
#define TEAL 0, 128, 128
#define ORANGE 255, 165, 0
#define ORCHID 218, 112, 214
#define BLACK 0, 0, 0
#define LIGHTSTEELBLUE 176, 196, 222
#define DARKORANGE 255, 140, 0
#define TOMATO 255, 99, 71

enum RequestMessageType {
  MessageType_InvalidRequest,
  MessageType_CreateOrAttachSession,
  MessageType_ServerInfo,
  MessageType_DeviceInfo,
  MessageType_ConnectPeer,
  MessageType_PeerHandshake,

  MessageType_CreateBuffer,
  MessageType_FreeBuffer,

  MessageType_CreateCommandQueue,
  MessageType_FreeCommandQueue,

  MessageType_CreateSampler,
  MessageType_FreeSampler,

  MessageType_CreateImage,
  MessageType_FreeImage,

  MessageType_CreateKernel,
  MessageType_FreeKernel,

  MessageType_BuildProgramFromSource,
  MessageType_BuildProgramFromBinary,
  MessageType_BuildProgramWithBuiltins,
  MessageType_BuildProgramWithDefinedBuiltins,
  // Special message type for SPIR-V IL for now. No support for
  // vendor-specific ILs.
  MessageType_BuildProgramFromSPIRV,
  MessageType_CompileProgramFromSPIRV,
  MessageType_CompileProgramFromSource,
  MessageType_LinkProgram,
  MessageType_FreeProgram,

  MessageType_CreateCommandBuffer,
  MessageType_FreeCommandBuffer,

  // ***********************************************

  MessageType_MigrateD2D,

  MessageType_Barrier,

  MessageType_ReadBuffer,
  MessageType_WriteBuffer,
  MessageType_CopyBuffer,
  MessageType_FillBuffer,

  MessageType_ReadBufferRect,
  MessageType_WriteBufferRect,
  MessageType_CopyBufferRect,

  MessageType_CopyImage2Buffer,
  MessageType_CopyBuffer2Image,
  MessageType_CopyImage2Image,
  MessageType_ReadImageRect,
  MessageType_WriteImageRect,
  MessageType_FillImageRect,

  MessageType_RunKernel,
  MessageType_RunCommandBuffer,

  MessageType_NotifyEvent,
  MessageType_RdmaBufferRegistration,

  // TODO finish
  MessageType_Finish,

  MessageType_Shutdown,
};

typedef struct {
  uint32_t Kind;
  uint64_t EID;
  uint64_t QID;
  uint64_t Submitted;
  uint64_t SendStart;
  uint64_t SendEnd;
  uint64_t RunStart;
  uint64_t RunEnd;
  uint64_t RecvStart;
  uint64_t RecvEnd;
  uint32_t Failed;
  size_t Lane;
} Event;

typedef struct {
  std::string Name;
  uint64_t FirstSubmitTS;
  uint64_t LastFinishTS;
  size_t NumLanes;
  std::vector<Event> Events;
} ServerData;

typedef struct {
  uint64_t FirstSubmitTS;
  uint64_t Duration;
  size_t TotalNumLanes;
  std::unordered_map<uint32_t, std::string> EventKindLabels;
  std::vector<ServerData> Servers;
} TraceFile;

const uint8_t MSGLABEL_ID = 1;
const uint8_t SERVERLABEL_ID = 2;
const uint8_t SERVER_START = 3;
const uint8_t SERVER_END = 4;
const uint8_t EVENT_ID = 6;

bool preprocessEvents(ServerData &Server) {
  std::cout << "Sorting events by submit timestamp..." << std::endl;
  std::sort(
      Server.Events.begin(), Server.Events.end(),
      [](const Event &A, const Event &B) { return A.Submitted < B.Submitted; });

  std::cout << "Spreading events into lanes to avoid overlap..." << std::endl;
  std::vector<uint64_t> LastFinishTSInLane{};
  for (Event &e : Server.Events) {
    bool Assigned = false;
    for (size_t Lane = 0; Lane < LastFinishTSInLane.size() && !Assigned;
         ++Lane) {
      if (LastFinishTSInLane[Lane] < e.Submitted) {
        e.Lane = Lane;
        LastFinishTSInLane[Lane] = e.RecvEnd;
        Assigned = true;
      }
    }
    if (!Assigned) {
      e.Lane = LastFinishTSInLane.size();
      LastFinishTSInLane.push_back(e.RecvEnd);
    }
  }
  Server.NumLanes = LastFinishTSInLane.size();
  return true;
}

bool readServer(std::istream &InStream, TraceFile &Data) {
  while (!InStream.eof()) {
    uint8_t Tag;
    InStream.read((char *)&Tag, sizeof(Tag));
    switch (Tag) {
    case SERVER_END: {
      return preprocessEvents(Data.Servers.back());
    }
    case SERVERLABEL_ID: {
      uint32_t Length;
      std::string Text;
      InStream.read((char *)&Length, sizeof(Length));
      Text.clear();
      Text.resize(Length);
      InStream.read(Text.data(), Length);
      Data.Servers.back().Name = Text;
      std::cout << "Read server label " << std::quoted(Text) << std::endl;
    } break;
    case EVENT_ID: {
      Event E{};
      ServerData &Server = Data.Servers.back();
      InStream.read((char *)&E.Kind, sizeof(E.Kind));
      InStream.read((char *)&E.QID, sizeof(E.QID));
      InStream.read((char *)&E.EID, sizeof(E.EID));
      InStream.read((char *)&E.Submitted, sizeof(E.Submitted));
      InStream.read((char *)&E.SendStart, sizeof(E.SendStart));
      InStream.read((char *)&E.SendEnd, sizeof(E.SendEnd));
      InStream.read((char *)&E.RunStart, sizeof(E.RunStart));
      InStream.read((char *)&E.RunEnd, sizeof(E.RunEnd));
      InStream.read((char *)&E.RecvStart, sizeof(E.RecvStart));
      InStream.read((char *)&E.RecvEnd, sizeof(E.RecvEnd));
      InStream.read((char *)&E.Failed, sizeof(E.Failed));

      if (E.Submitted < Server.FirstSubmitTS)
        Server.FirstSubmitTS = E.Submitted;
      if (E.RecvEnd > Server.LastFinishTS)
        Server.LastFinishTS = E.RecvEnd;

      Data.Servers.back().Events.push_back(E);
    } break;
    case MSGLABEL_ID:
    case SERVER_START:
      std::cerr << "Invalid tag while reading server: " << Tag << std::endl;
      return false;
      break;
    default:
      std::cerr << "Unknown tag: " << Tag << std::endl;
      return false;
    }
  }

  std::cerr << "Unexpected EOF while reading server data" << std::endl;
  return false;
}

bool readTrace(const std::filesystem::path &File, TraceFile &Data) {
  std::ifstream InStream(File, std::ios::binary);
  if (!InStream.is_open()) {
    std::cerr << "Failed to open " << File << std::endl;
    return false;
  }
  std::array<char, 8> Magic{};
  std::array<char, 8> ReferenceMagic{'P', 'O', 'C', 'L', 'R', 'T', 'R', 'C'};
  InStream.read(Magic.data(), Magic.size());
  if (Magic != ReferenceMagic) {
    std::cerr << "Wrong magic bytes -- this does not look like a pocl-remote "
                 "trace file"
              << std::endl;
    return false;
  }

  Data.EventKindLabels.clear();
  Data.Servers.clear();
  while (!InStream.eof()) {
    uint8_t Tag;
    InStream.read((char *)&Tag, sizeof(Tag));
    if (InStream.eof())
      break;
    switch (Tag) {
    case MSGLABEL_ID: {
      uint32_t Kind;
      uint32_t Length;
      std::string Text;
      InStream.read((char *)&Kind, sizeof(Kind));
      InStream.read((char *)&Length, sizeof(Length));
      Text.clear();
      Text.resize(Length);
      InStream.read(Text.data(), Length);
      Data.EventKindLabels[Kind] = Text;
      std::cout << "Read event kind label (" << Kind << ", "
                << std::quoted(Text) << ")" << std::endl;
    } break;
    case SERVER_START: {
      Data.Servers.push_back(ServerData());
      Data.Servers.back().FirstSubmitTS = UINT64_MAX;
      Data.Servers.back().LastFinishTS = 0;
      if (!readServer(InStream, Data)) {
        Data.Servers.pop_back();
        return false;
      }
      Data.TotalNumLanes += Data.Servers.back().NumLanes;
      std::cout << "Server " << Data.Servers.back().Name << " has "
                << Data.Servers.back().Events.size() << " events across "
                << Data.Servers.back().NumLanes << " lanes." << std::endl;
    } break;
    case SERVER_END:
    case SERVERLABEL_ID:
    case EVENT_ID:
      std::cerr << "Invalid tag at top level: " << Tag << std::endl;
      return false;
      break;
    default:
      std::cerr << "Unknown tag: " << Tag << std::endl;
      return false;
    }
  }
  uint64_t LastFinishTS = 0;
  Data.FirstSubmitTS = UINT64_MAX;
  for (const ServerData &S : Data.Servers) {
    Data.FirstSubmitTS = std::min(Data.FirstSubmitTS, S.FirstSubmitTS);
    LastFinishTS = std::max(LastFinishTS, S.LastFinishTS);
  }
  Data.Duration = LastFinishTS - Data.FirstSubmitTS;

  return true;
}

int drawLegend(SDL_Renderer *Renderer, float OffsetX, float OffsetY) {
  int Ok = 1;
  float Advance = 0.0f;
  SDL_FRect Rect;
#define ADD_LABEL(label, color)                                                \
  do {                                                                         \
    Rect.x = OffsetX + Advance;                                                \
    Rect.y = OffsetY;                                                          \
    Rect.w = 12;                                                               \
    Rect.h = 12;                                                               \
    Ok &= SDL_SetRenderDrawColor(Renderer, BLACK, SDL_ALPHA_OPAQUE);           \
    Ok &= SDL_RenderFillRect(Renderer, &Rect);                                 \
    Rect.x += 1;                                                               \
    Rect.y += 1;                                                               \
    Rect.w -= 2;                                                               \
    Rect.h -= 2;                                                               \
    Ok &= SDL_SetRenderDrawColor(Renderer, color, SDL_ALPHA_OPAQUE);           \
    Ok &= SDL_RenderFillRect(Renderer, &Rect);                                 \
    Ok &= SDL_SetRenderDrawColor(Renderer, BLACK, SDL_ALPHA_OPAQUE);           \
    Ok &= SDL_RenderDebugText(Renderer, Rect.x + 16, Rect.y + 2, label);       \
    Advance += strlen(label) * 8 + 24;                                         \
  } while (0)

  ADD_LABEL("Sending", GREEN);
  ADD_LABEL("Copying", ORCHID);
  ADD_LABEL("Running", ORANGE);
  ADD_LABEL("Receiving", TEAL);
  ADD_LABEL("Failed", TOMATO);

#undef ADD_LABEL
  return Ok;
}

void computeNewMinimap(const TraceFile &Data, float Width,
                       std::vector<SDL_FRect> &OutChunks) {
  std::vector<SDL_FRect> AllChunks;
  float OffsetY = 4;
  for (const ServerData &S : Data.Servers) {
    std::vector<std::vector<SDL_FRect>> Chunks(S.NumLanes);
    for (const Event &E : S.Events) {
      float EventWidth =
          ((E.RecvEnd - E.Submitted) / (double)(Data.Duration)) * Width;
      float EventStartTS =
          ((E.Submitted - Data.FirstSubmitTS) / (double)(Data.Duration)) *
          Width;
      SDL_FRect Ev{std::round(EventStartTS), OffsetY + 3 * E.Lane,
                   std::max(EventWidth, 3.0f), 4};

      size_t FirstIdxAfter = 0;
      bool Merged = false;
      for (SDL_FRect &Chunk : Chunks[E.Lane]) {
        if (Ev.x <= Chunk.x && Ev.x + Ev.w >= Chunk.x + Chunk.w) {
          Merged = true;
        } else if (Ev.x >= Chunk.x && Ev.x + Ev.w <= Chunk.x + Chunk.w) {
          Merged = true;
        } else if (Ev.x >= Chunk.x && Ev.x <= Chunk.x + Chunk.w) {
          Merged = true;
        } else if (Ev.x + Ev.w >= Chunk.x && Ev.x + Ev.w <= Chunk.x + Chunk.w) {
          Merged = true;
        }
        if (Merged) {
          float start = std::min(Ev.x, Chunk.x);
          float len = std::max(Ev.x + Ev.w, Chunk.x + Chunk.w) - start;
          Chunk.x = start;
          Chunk.w = len;
          break;
        }
        if (Chunk.x + Chunk.w < Ev.x) {
          FirstIdxAfter++;
        }
      }
      if (!Merged) {
        Chunks[E.Lane].insert(Chunks[E.Lane].begin() + FirstIdxAfter, Ev);
      }
    }
    for (std::vector<SDL_FRect> &lane : Chunks) {
      AllChunks.insert(AllChunks.end(), lane.begin(), lane.end());
    }
    OffsetY += 3 * S.NumLanes;
  }

  std::cout << "Minimap has " << AllChunks.size() << " chunks" << std::endl;
  OutChunks = AllChunks;
}

int createMinimapTexture(SDL_Renderer *Renderer, SDL_Texture **Texture,
                         const TraceFile &Trace, const SDL_FRect &WindowRect) {
  int Ok = 1;
  SDL_FRect Rect{0, 0, WindowRect.w, WindowRect.h};
  std::vector<SDL_FRect> Chunks;
  SDL_Texture *NewTexture =
      SDL_CreateTexture(Renderer, SDL_PIXELFORMAT_RGBA32,
                        SDL_TEXTUREACCESS_TARGET, WindowRect.w, MINIMAP_HEIGHT);
  if (!NewTexture)
    return 0;
  computeNewMinimap(Trace, WindowRect.w, Chunks);

  Ok &= SDL_SetRenderTarget(Renderer, NewTexture);
  Ok &= SDL_SetRenderDrawColor(Renderer, LIGHTSTEELBLUE, SDL_ALPHA_OPAQUE);
  Ok &= SDL_RenderFillRect(Renderer, &Rect);
  Ok &= SDL_SetRenderDrawColor(Renderer, BLACK, SDL_ALPHA_OPAQUE);
  Ok &= SDL_RenderFillRects(Renderer, Chunks.data(), Chunks.size());
  for (SDL_FRect &R : Chunks) {
    R.x += 1;
    R.y += 1;
    R.w -= 2;
    R.h -= 2;
  }
  Ok &= SDL_SetRenderDrawColor(Renderer, DARKGREY, SDL_ALPHA_OPAQUE);
  Ok &= SDL_RenderFillRects(Renderer, Chunks.data(), Chunks.size());

  if (!Ok)
    SDL_DestroyTexture(NewTexture);
  else if (*Texture)
    SDL_DestroyTexture(*Texture);
  *Texture = NewTexture;
  Ok &= SDL_SetRenderTarget(Renderer, NULL);
  return Ok;
}

int drawMinimap(SDL_Renderer *Renderer, float OffsetY, float Width,
                float Height, float VisibleStartPct, float ViewWidth,
                SDL_Texture *MapImage) {
  int Ok = 1;
  SDL_FRect Rect{0, OffsetY, Width, Height};
  Ok &= SDL_RenderTexture(Renderer, MapImage, NULL, &Rect);

  Rect.x = Width * VisibleStartPct;
  Rect.w = std::max(Width * ViewWidth, 1.0f);
  Ok &= SDL_SetRenderDrawColor(Renderer, DARKORANGE, SDL_ALPHA_OPAQUE * 0.2);
  Ok &= SDL_RenderFillRect(Renderer, &Rect);
  SDL_FPoint LinePoints[] = {
      {Rect.x, Rect.y},
      {Rect.x + Rect.w, Rect.y},
      {Rect.x + Rect.w, Rect.y + Rect.h - 1},
      {Rect.x, Rect.y + Rect.h - 1},
      {Rect.x, Rect.y},
  };
  Ok &= SDL_SetRenderDrawColor(Renderer, DARKORANGE, SDL_ALPHA_OPAQUE);
  Ok &= SDL_RenderLines(Renderer, LinePoints,
                        sizeof(LinePoints) / sizeof(SDL_FPoint));
  return Ok;
}

int drawEventView(SDL_Renderer *Renderer, float OffsetY, float Width,
                  float Height, float ScrollY, uint64_t StartTS, uint64_t EndTS,
                  const TraceFile &Trace) {
  int Ok = 1;
  uint64_t ViewDuration = EndTS - StartTS;
  for (const ServerData &S : Trace.Servers) {
    SDL_SetRenderDrawColor(Renderer, BLACK, SDL_ALPHA_OPAQUE);
    SDL_RenderDebugText(Renderer, 0, OffsetY, S.Name.c_str());
    OffsetY += 12;
    SDL_Rect EventClipRect{0, (int)OffsetY, (int)Width, (int)Height - 12};
    SDL_SetRenderClipRect(Renderer, &EventClipRect);
    for (const Event &E : S.Events) {
      if (E.RecvEnd >= StartTS && E.Submitted <= EndTS) {
        SDL_FRect R;
        if (E.Submitted >= StartTS)
          R.x = ((E.Submitted - StartTS) / (double)ViewDuration) * Width;
        else
          R.x = -((StartTS - E.Submitted) / (double)ViewDuration) * Width;
        R.y = OffsetY + 12 * E.Lane - ScrollY;
        R.w = std::max(
            3.0f,
            (float)((E.RecvEnd - E.Submitted) / (double)ViewDuration) * Width);
        R.h = EVENTVIEW_EVENT_HEIGHT;
        SDL_SetRenderDrawColor(Renderer, BLACK, SDL_ALPHA_OPAQUE);
        SDL_RenderFillRect(Renderer, &R);
        SDL_Rect cr{(int)R.x, std::max((int)R.y, EventClipRect.y), (int)R.w,
                    (int)R.h};
        cr.h = std::max(0, std::min(cr.h, (int)(R.y - OffsetY + 12)));
        if (cr.h == 0)
          continue;
        SDL_SetRenderClipRect(Renderer, &cr);
        R.x += 1;
        R.y += 1;
        R.w -= 2;
        R.h -= 2;
        if (E.Failed)
          SDL_SetRenderDrawColor(Renderer, TOMATO, SDL_ALPHA_OPAQUE);
        else
          SDL_SetRenderDrawColor(Renderer, DARKGREY, SDL_ALPHA_OPAQUE);
        SDL_RenderFillRect(Renderer, &R);

#define WRITE_HIGHLIGHT(name, fillcolor)                                       \
  do {                                                                         \
    SDL_FRect Rect{R.x, R.y, R.w, R.h};                                        \
    Rect.x =                                                                   \
        R.x + ((E.name##Start - E.Submitted) / (double)ViewDuration) * Width;  \
    Rect.x = std::min(Rect.x, R.x + R.w - 1.0f);                               \
    int64_t Length = (E.name##End - E.name##Start);                            \
    Rect.w = std::max((((double)Length) / (double)ViewDuration) * Width, 1.0); \
    if (E.name##End == 0 || Length <= 0) {                                     \
      break; /* skip */                                                        \
    }                                                                          \
    SDL_SetRenderDrawColor(Renderer, fillcolor, SDL_ALPHA_OPAQUE);             \
    SDL_RenderFillRect(Renderer, &Rect);                                       \
  } while (0)

        WRITE_HIGHLIGHT(Send, GREEN);
        switch (E.Kind) {
        case MessageType_ServerInfo:
        case MessageType_DeviceInfo:
        case MessageType_MigrateD2D:
        case MessageType_ReadBuffer:
        case MessageType_WriteBuffer:
        case MessageType_CopyBuffer:
        case MessageType_FillBuffer:
        case MessageType_ReadBufferRect:
        case MessageType_WriteBufferRect:
        case MessageType_CopyBufferRect:
        case MessageType_CopyImage2Buffer:
        case MessageType_CopyBuffer2Image:
        case MessageType_CopyImage2Image:
        case MessageType_ReadImageRect:
        case MessageType_WriteImageRect:
        case MessageType_FillImageRect:
          WRITE_HIGHLIGHT(Run, ORCHID);
          break;
        default:
          WRITE_HIGHLIGHT(Run, ORANGE);
        }
        WRITE_HIGHLIGHT(Recv, TEAL);

#undef WRITE_HIGHLIGHT

        SDL_SetRenderDrawColor(Renderer, BLACK, SDL_ALPHA_OPAQUE);
        if (E.QID == UINT64_MAX) {
          SDL_RenderDebugText(Renderer, R.x + 1, R.y + 1,
                              Trace.EventKindLabels.at(E.Kind).c_str());
        } else {
          SDL_RenderDebugTextFormat(Renderer, R.x + 1, R.y + 1,
                                    "%s Q:%lu/E:%lu",
                                    Trace.EventKindLabels.at(E.Kind).c_str(),
                                    (unsigned long)E.QID, (unsigned long)E.EID);
        }
        SDL_SetRenderClipRect(Renderer, &EventClipRect);
      }
    }
    OffsetY += EVENTVIEW_EVENT_HEIGHT * (S.NumLanes + 1);
  }

  SDL_SetRenderClipRect(Renderer, nullptr);
  return Ok;
}

int main(int Argc, char **Argv) {
  if (Argc < 2) {
    std::cerr << "Usage: " << Argv[0] << " [filename]" << std::endl;
    return 1;
  }
  if (!SDL_Init(SDL_INIT_EVENTS | SDL_INIT_VIDEO)) {
    std::cerr << "SDL_Init: " << SDL_GetError() << std::endl;
    return 1;
  }

  TraceFile Trace{};
  if (!readTrace(Argv[1], Trace)) {
    return 1;
  }

  SDL_Window *Window;
  SDL_Renderer *Renderer;
  SDL_FRect WindowRect = {0, 0, INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT};
  if (!SDL_CreateWindowAndRenderer("PoCL-Remote trace viewer",
                                   INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT,
                                   SDL_WINDOW_RESIZABLE, &Window, &Renderer)) {
    std::cerr << "SDL_CreateWindowAndRenderer: " << SDL_GetError() << std::endl;
    return 1;
  }
  SDL_SetRenderDrawBlendMode(Renderer, SDL_BLENDMODE_BLEND);
  SDL_ShowWindow(Window);

  SDL_Texture *MinimapImage = nullptr;
  bool NeedNewMinimap = true;

  uint64_t ViewTargetTS = 0;
  float TimeScale = 1e4;
  float EventsScrollY = 0;
  uint64_t ViewDuration = TimeScale * WindowRect.w;
  uint64_t StartTS = 0;
  uint64_t EndTS = StartTS + ViewDuration;
  int Ok = 1;
  bool Done = false;
  int ShiftIsHeld = false;
  int CtrlIsHeld = false;
  while (Ok && !Done) {
    SDL_Event Ev;
    if (!SDL_WaitEvent(nullptr)) {
      std::cerr << "SDL_WaitEvent: " << SDL_GetError() << std::endl;
      return 1;
    }
    while (SDL_PollEvent(&Ev)) {
      if (Ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED ||
          Ev.type == SDL_EVENT_QUIT) {
        Done = true;
        break;
      } else if (Ev.type == SDL_EVENT_KEY_DOWN) {
        if (Ev.key.key == SDLK_ESCAPE) {
          Done = true;
          break;
        } else if (Ev.key.key == SDLK_LSHIFT) {
          ShiftIsHeld = Ev.key.down;
        } else if (Ev.key.key == SDLK_LCTRL) {
          CtrlIsHeld = Ev.key.down;
        }
      } else if (Ev.type == SDL_EVENT_KEY_UP) {
        if (Ev.key.key == SDLK_LSHIFT) {
          ShiftIsHeld = Ev.key.down;
        } else if (Ev.key.key == SDLK_LCTRL) {
          CtrlIsHeld = Ev.key.down;
        }
      } else if (Ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
                 Ev.button.button == SDL_BUTTON_LEFT) {
        if (Ev.button.y >= MINIMAP_Y &&
            Ev.button.y <= MINIMAP_Y + MINIMAP_HEIGHT) {
          ViewTargetTS =
              (uint64_t)((double)(Ev.button.x / WindowRect.w) * Trace.Duration);
          StartTS = ViewTargetTS;
          EndTS = StartTS + ViewDuration;
        }
      } else if (Ev.type == SDL_EVENT_MOUSE_WHEEL) {
        if (ShiftIsHeld) {
          EventsScrollY =
              std::max(0.0f, EventsScrollY -
                                 Ev.wheel.y * EVENTVIEW_EVENT_HEIGHT * 10.0f);
          EventsScrollY = std::min(
              EventsScrollY,
              std::max(0.0f,
                       Trace.TotalNumLanes * EVENTVIEW_EVENT_HEIGHT -
                           (WindowRect.h - MINIMAP_Y - MINIMAP_HEIGHT - 20)));
          ;
        } else if (CtrlIsHeld) {
          TimeScale = std::max(1.0f, TimeScale + Ev.wheel.y * (TimeScale / 10));
          TimeScale = std::min(TimeScale, Trace.Duration / WindowRect.w);
        } else {
          int64_t Shift = WindowRect.w * Ev.wheel.y * TimeScale / 10.0f;
          if (Shift > 0)
            StartTS -= std::min((uint64_t)Shift, StartTS);
          else
            StartTS = std::min(StartTS - Shift, Trace.Duration - ViewDuration);
        }
      } else if (Ev.type == SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED)
        std::cerr << "TODO: SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED"
                  << std::endl;
      else if (Ev.type == SDL_EVENT_WINDOW_RESIZED |
               Ev.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
        if (Ev.window.data1 > 0 && Ev.window.data2 > 0) {
          WindowRect.w = (float)Ev.window.data1;
          WindowRect.h = (float)Ev.window.data2;
          NeedNewMinimap = true;
        }
      }
    }

    ViewDuration = TimeScale * WindowRect.w;
    EndTS = StartTS + ViewDuration;
    if (NeedNewMinimap) {
      Ok &= createMinimapTexture(Renderer, &MinimapImage, Trace, WindowRect);
      NeedNewMinimap = false;
    }

    Ok &= SDL_SetRenderDrawColor(Renderer, ALICEBLUE, SDL_ALPHA_OPAQUE);
    Ok &= SDL_RenderClear(Renderer);
    Ok &= SDL_SetRenderDrawColor(Renderer, BLACK, SDL_ALPHA_OPAQUE);
    Ok &= SDL_RenderDebugText(Renderer, WindowRect.w - strlen(Argv[1]) * 8 - 8,
                              11, Argv[1]);

    Ok &= drawLegend(Renderer, 8, 8);
    Ok &= drawMinimap(Renderer, MINIMAP_Y, WindowRect.w, MINIMAP_HEIGHT,
                      StartTS / (double)Trace.Duration,
                      (EndTS - StartTS) / (double)Trace.Duration, MinimapImage);
    Ok &= drawEventView(Renderer, MINIMAP_Y + MINIMAP_HEIGHT + 8, WindowRect.w,
                        WindowRect.h - MINIMAP_Y + MINIMAP_HEIGHT + 8,
                        EventsScrollY, StartTS + Trace.FirstSubmitTS,
                        EndTS + Trace.FirstSubmitTS, Trace);

    Ok &= SDL_RenderPresent(Renderer);
  }
  if (!Ok)
    std::cerr << "Renderer error: " << SDL_GetError() << std::endl;

  if (MinimapImage)
    SDL_DestroyTexture(MinimapImage);
  SDL_DestroyRenderer(Renderer);
  SDL_DestroyWindow(Window);
}
