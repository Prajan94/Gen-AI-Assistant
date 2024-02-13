
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { ReactiveFormsModule } from '@angular/forms';
import { FormsModule } from '@angular/forms';
import { HttpClientModule }  
    from '@angular/common/http'; 

/* App Root */
import { AppComponent } from './app.component';
import { YoutubeTranscriptAIComponent } from './youtube-transcript-ai/youtube-transcript-ai.component';
import { EmailQueryComponent } from './email-query/email-query.component';
import { FigmaCodeGeneratorComponent } from './figma-code-generator/figma-code-generator.component';
import { SyntheticDataGenComponent } from './synthetic-data-gen/synthetic-data-gen.component';
import { HomeComponent } from './home/home.component';
import { SpeechToTextComponent } from './speech-to-text/speech-to-text.component';
/* Imports for Material UI */
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import {MatInputModule} from '@angular/material/input';
import {MatChipsModule} from '@angular/material/chips';
import {MatButtonModule} from '@angular/material/button';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';

/* Routing Module */
import { AppRoutingModule } from './app.routes';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';

@NgModule({
  imports: [
    BrowserModule,
    AppRoutingModule,
    ReactiveFormsModule,
    FormsModule,
    BrowserAnimationsModule,
    HttpClientModule,
    MatSlideToggleModule,
    MatInputModule,
    MatChipsModule,
    MatButtonModule,
    MatProgressBarModule,
    MatProgressSpinnerModule
  ],
  declarations: [
    AppComponent,
    YoutubeTranscriptAIComponent,
    EmailQueryComponent,
    FigmaCodeGeneratorComponent,
    SyntheticDataGenComponent,
    HomeComponent,
    SpeechToTextComponent
  ],
  bootstrap: [AppComponent],
  providers: [
    provideAnimationsAsync()
  ]
})
export class AppModule { }


/*
Copyright Google LLC. All Rights Reserved.
Use of this source code is governed by an MIT-style license that
can be found in the LICENSE file at https://angular.io/license
*/