import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonServiceService } from '../common-service.service';
import { Router } from '@angular/router';
import RecordRTC from 'recordrtc';
import { DomSanitizer } from '@angular/platform-browser';
import { saveAs, FileSaverOptions } from 'file-saver';

const type = {
  type: 'audio',
  mimeType: 'audio/mp3',
  timeSlice: 10000,
};

@Component({
  selector: 'app-speech-to-text',
  templateUrl: './speech-to-text.component.html',
  styleUrl: './speech-to-text.component.scss'
})

export class SpeechToTextComponent {
  response: any;
  aiAuery: any;
  hideBotMessage = false;
  isLoading = false;
  isRecording = false;
  aiMessageArray: any = [];
  humanMessageArray:any = [];  
  recorder: any;
  recordedChunks: Array<any> = [];
  stream: any;
  constructor(
    private commonService : CommonServiceService, private route: Router, private sanitizer: DomSanitizer, private ref: ChangeDetectorRef) { }
    askAi(blob: any) {
      // this.aiMessageArray.push(this.aiAuery);
      this.commonService.voiceAssistant(blob).subscribe(item => {
      this.aiMessageArray.push(item[1]);
      this.response = item;
      this.humanMessageArray.push(item[0]);
      this.isLoading = false;
      this.ref.detectChanges();
    }, ((error: any) => {
      if(error.status == 200) {
        this.response = error.error.text;
        this.sanitizer.bypassSecurityTrustStyle(this.response);
        this.isLoading = false;
      } else {
        this.isLoading = false;
        console.log("Error from flask server --->" + error);
        this.response = "Network Error occured, please try again";
      }
    }))
  }

  gotoHome() {
    this.route.navigateByUrl("home");
  }

  onMicClick() {
    this.isRecording = !this.isRecording
    if(this.isRecording) {
      navigator.mediaDevices.getUserMedia({
        audio: true
    }).then(async (stream) => {
        this.recorder = new RecordRTC(stream, {
            type: 'audio'
        });
        this.recorder.startRecording();
    
        // const sleep = (m: number | undefined) => new Promise(r => setTimeout(r, m));
        // await sleep(10000);
    });
    } else {
      this.stopRercording();
    }
    // this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // this.recorder = new RecordRTC(this.stream, {type: 'audio/wav'});
    // this.recorder.ondataavailable = (event: { data: any; }) => {
    //   this.recordedChunks.push(event.data)
    // }
    // this.recorder = new MediaRecorder(this.stream, {
    //   type: 'audio', mimeType: 'audio/wav',
    //   timeSlice: 3000,
    //   ondataavailable: (blob) => {
    //     this.recordedChunks.push(blob);
    //     this.ref.detectChanges();
    //   },
    // });
    // this.recorder.start();
  }
  stopRercording() {
    this.hideBotMessage = true;
    this.isLoading = true;
    this.recorder.stopRecording(() => {
      let blob = this.recorder.getBlob();
      // saveAs(blob, "audio.webm");
      this.askAi(blob);
  });
    // this.recorder.stop();
    // let blob = new Blob(this.recordedChunks, { type: this.recorder.mimeType });
    // this.askAi(blob);
    // this.recorder.stop(() => {
    //   this.ref.detectChanges();
    //   debugger;
    //   let blob = new Blob(this.recordedChunks, { type: this.recorder.mimeType });
    //   this.askAi(blob);
    // });

  }

}
