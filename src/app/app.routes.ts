import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { YoutubeTranscriptAIComponent } from './youtube-transcript-ai/youtube-transcript-ai.component';
import { EmailQueryComponent } from './email-query/email-query.component';
import { FigmaCodeGeneratorComponent } from './figma-code-generator/figma-code-generator.component';
import { SyntheticDataGenComponent } from './synthetic-data-gen/synthetic-data-gen.component';
import { HomeComponent } from './home/home.component';
import { SpeechToTextComponent } from './speech-to-text/speech-to-text.component';
import { ImageToHtmlComponent } from './image-to-html/image-to-html.component';
import { TestCaseGeneratorAIComponent } from './test-case-generator-ai/test-case-generator-ai.component';
import { PdfChatBotComponent } from './pdf-chat-bot/pdf-chat-bot.component';
import { ServerLogAssistantComponent } from './server-log-assistant/server-log-assistant.component';

export const routes: Routes = [
  {path: "home", component: HomeComponent},
  { path: "transAI", component: YoutubeTranscriptAIComponent},
  {path:"emailAI", component: EmailQueryComponent},
  {path:"figmaAI", component: FigmaCodeGeneratorComponent},
  {path: "synGenAI", component: SyntheticDataGenComponent},
  {path: "speechAI", component: SpeechToTextComponent},
  {path: "htmlAI", component: ImageToHtmlComponent},
  {path: "unitTestAI", component: TestCaseGeneratorAIComponent},
  {path: "pdfChatBot", component: PdfChatBotComponent},
  {path: "serverLogAss", component: ServerLogAssistantComponent}

]

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}