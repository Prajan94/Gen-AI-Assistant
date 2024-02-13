import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonServiceService } from '../common-service.service';
import { Router } from '@angular/router';
import RecordRTC from 'recordrtc';
// import * as RecordRTC from '../../assets/RecordRTC.js';
// import RecordRTC from 'asset/recordrtc.js';
import { DomSanitizer } from '@angular/platform-browser';

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
      this.hideBotMessage = true;
      this.isLoading = true;
      this.aiMessageArray.push(this.aiAuery);
      this.commonService.voiceAssistant(blob).subscribe(item => {
      this.response = item;
      this.humanMessageArray.push(this.response);
      this.isLoading = false;
    }, ((error: any) => {
      this.isLoading = false;
      console.log("Error from flask server --->" + error);
      this.response = "Network Error occured, please try again";
    }))
  }

  gotoHome() {
    this.route.navigateByUrl("home");
  }

  // onMicClick() {
  //   this.isRecording = !this.isRecording;
  //   if (this.isRecording) {
  //     this.startRercording();
  //   } else {
  //     this.stopRercording();
  //   }
  // }
  onMicClick() {
    navigator.mediaDevices.getUserMedia({
      audio: true
  }).then(async (stream) => {
      let recorder = new RecordRTC(stream, {
          type: 'audio'
      });
      recorder.startRecording();
  
      const sleep = (m: number | undefined) => new Promise(r => setTimeout(r, m));
      await sleep(3000);
  
      recorder.stopRecording(() => {
          let blob = recorder.getBlob();
          this.askAi(blob);
      });
  });
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
    this.recorder.stop();
    let blob = new Blob(this.recordedChunks, { type: this.recorder.mimeType });
    this.askAi(blob);
    // this.recorder.stop(() => {
    //   this.ref.detectChanges();
    //   debugger;
    //   let blob = new Blob(this.recordedChunks, { type: this.recorder.mimeType });
    //   this.askAi(blob);
    // });

  }

}
