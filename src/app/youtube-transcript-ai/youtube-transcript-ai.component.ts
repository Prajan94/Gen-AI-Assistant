import { Component } from '@angular/core';
import { CommonServiceService } from '../common-service.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-youtube-transcript-ai',
  templateUrl: './youtube-transcript-ai.component.html',
  styleUrl: './youtube-transcript-ai.component.scss'
})
export class YoutubeTranscriptAIComponent {
  response: any;
  aiAuery: any;
  course: any;
  isLoading = false;
  hideBotMessage = false;
  errMsg: any;
  aiMessageArray: any = [];
  humanMessageArray:any = [];
  constructor(
    private commonService : CommonServiceService, private route: Router) {
     }
    askAi() {
      if (this.course) {
        this.errMsg = "";
        this.hideBotMessage = true;
        this.isLoading = true;
        this.aiMessageArray.push(this.aiAuery);
        this.commonService.askAI(this.aiAuery,this.course).subscribe(item => {
        this.response = item;
        this.humanMessageArray.push(item);
        this.isLoading = false;
      }, ((error: any) => {
        this.isLoading = false;
        console.log("Error from flask server --->" + error);
        this.response = "Network Error occured, please try again";
      }))
      } else {
        this.errMsg = "Please select course";
      }
  }
  gotoHome() {
    this.route.navigateByUrl("home");
  }
}
