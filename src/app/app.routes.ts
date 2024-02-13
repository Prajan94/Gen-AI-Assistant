import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { YoutubeTranscriptAIComponent } from './youtube-transcript-ai/youtube-transcript-ai.component';
import { EmailQueryComponent } from './email-query/email-query.component';
import { FigmaCodeGeneratorComponent } from './figma-code-generator/figma-code-generator.component';
import { SyntheticDataGenComponent } from './synthetic-data-gen/synthetic-data-gen.component';
import { HomeComponent } from './home/home.component';
import { SpeechToTextComponent } from './speech-to-text/speech-to-text.component';

export const routes: Routes = [
  {path: "home", component: HomeComponent},
  { path: "transAI", component: YoutubeTranscriptAIComponent},
  {path:"emailAI", component: EmailQueryComponent},
  {path:"figmaAI", component: FigmaCodeGeneratorComponent},
  {path: "synGenAI", component: SyntheticDataGenComponent},
  {path: "speechAI", component: SpeechToTextComponent}

]

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}